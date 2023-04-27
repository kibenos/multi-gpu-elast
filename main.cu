#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <type_traits>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "json.hpp"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_2D_X 32
#define BLOCK_SIZE_2D_Y 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cout << "[" + std::string(file) + ":" + std::to_string(line) + "] " + "CUDA error: " + std::string(cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}


struct texture_wrapper_t
{
    cudaTextureObject_t tex_obj;
    cudaArray_t         arr_obj;
    size_t              dev_idx;

    static texture_wrapper_t make_texture_wrapper(cudaTextureObject_t obj, cudaArray_t arr, size_t device_idx)
    {
        return texture_wrapper_t{ obj, arr, device_idx };
    }
};


struct amax_t : public thrust::binary_function<double, double, bool>
{
    __host__ __device__
        bool operator()(double lhs, double rhs)
    {
        return abs(lhs) < abs(rhs);
    }
};


template <class T>
texture_wrapper_t set_tex(
    T* host_data, 
    size_t xsz, 
    size_t ysz, 
    size_t zsz = 1, 
    cudaMemcpyKind mcpy_kind = cudaMemcpyHostToDevice,
    cudaTextureAddressMode adr_mode = cudaAddressModeBorder
)
{
    // allocate CUDA 3D array in device memory
    auto channel_desc = cudaCreateChannelDesc<T>();
    auto extent = make_cudaExtent(xsz, ysz, zsz);
    cudaArray_t cu_array;
    gpuErrchk(cudaMalloc3DArray(&cu_array, &channel_desc, extent));

    // copy data to device
    size_t pitch = xsz * sizeof(T);
    auto cu_ptr = make_cudaPitchedPtr(host_data, pitch, xsz, ysz);
    cudaMemcpy3DParms copy3d_parms = { 0 };
    copy3d_parms.srcPtr = cu_ptr;
    copy3d_parms.dstArray = cu_array;
    copy3d_parms.extent = extent;
    copy3d_parms.kind = mcpy_kind;
    gpuErrchk(cudaMemcpy3D(&copy3d_parms));

    // specify texture
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    // specify texture object parameters
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = adr_mode;
    tex_desc.addressMode[1] = adr_mode;
    tex_desc.addressMode[2] = adr_mode;

    if (std::is_same<T, float>::value)
        tex_desc.filterMode = cudaFilterModeLinear;
    else
        tex_desc.filterMode = cudaFilterModePoint;

    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    // create texture object
    cudaTextureObject_t tex_obj = 0;
    gpuErrchk(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL));

    int device_idx;
    gpuErrchk(cudaGetDevice(&device_idx));

    return texture_wrapper_t::make_texture_wrapper(tex_obj, cu_array, device_idx);
}


template <class T>
T* read_file(std::string const& fname, size_t size)
{
    std::ifstream file(fname, std::ios::binary);
    if (!file) { std::cout << "cannot open file: " << fname << '\n'; exit(-1); }

    T* data = new T[size];
    file.read(reinterpret_cast<char*>(data), size * sizeof(T));

    return data;
}


template <class T>
void set_matrix_zero(T** arr_dev, size_t m, size_t n) {
    gpuErrchk(cudaMalloc(arr_dev, m * n * sizeof(T)));
    gpuErrchk(cudaMemset(*arr_dev, 0, m * n * sizeof(T)));
}


void save_dev_arr(void const* src_dev, std::string const& filename, void* buff_host, size_t size) {
    gpuErrchk(cudaMemcpy(buff_host, src_dev, size, cudaMemcpyDeviceToHost));
    std::ofstream file(filename, std::ios_base::binary);
    if (!file) { std::cout << "cannot open file " << filename << "\n"; exit(-1); }
    file.write(static_cast<char const*>(buff_host), size);
}


double amax(double const* arr_dev, size_t size) {
    double maxel;
    thrust::device_ptr<double const> const ptr_dev = thrust::device_pointer_cast(arr_dev);
    size_t maxel_pos = thrust::max_element(thrust::device, ptr_dev, ptr_dev + size, amax_t()) - ptr_dev;
    gpuErrchk(cudaMemcpy(&maxel, arr_dev + maxel_pos, sizeof(double), cudaMemcpyDeviceToHost));

    return std::abs(maxel);
}


struct kernel_params_t
{
    // space arrays
    double* P, * tauXX, * tauYY, * tauXY;     // stress
    double* Ux, * Uy;                         // displacement
    double* Vx, * Vy;                         // velocity

    // materials
    cudaTextureObject_t mdata, K, G;

    // input parameters
    double dt;
    double dX, dY;
    double Lx, Ly;
    double dampX, dampY;
    double rho0;
    double coh;
    double P0;
    size_t Nx, Ny;

    // size of the slice along the y-axis
    size_t NyS;

    // shift of the slice with respect to the origin of the mesh along the y-axis
    size_t yshift;
};


__global__ void SetDisp(double dUxdx, double dUydy, double dUxdy, kernel_params_t const pa)
{
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t const y = pa.yshift + j;

    size_t const Nx = pa.Nx, Ny = pa.Ny;
    const double dX = pa.dX, dY = pa.dY;
    double* const Ux = pa.Ux;
    double* const Uy = pa.Uy;

    if (i < Nx + 1 && j < pa.NyS)
        Ux[j * (Nx + 1) + i] = (-0.5 * dX * Nx + dX * i) * dUxdx + (-0.5 * dY * (Ny - 1) + dY * y) * dUxdy;
    if (i < Nx && j < pa.NyS + 1)
        Uy[j * Nx + i] = (-0.5 * dY * Ny + dY * y) * dUydy;
}


__global__ void ComputeDisp(kernel_params_t const pa)
{
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

    double* const Ux = pa.Ux;
    double* const Uy = pa.Uy;
    double* const Vx = pa.Vx;
    double* const Vy = pa.Vy;
    double const* const P = pa.P;
    double const* const tauXX = pa.tauXX;
    double const* const tauYY = pa.tauYY;
    double const* const tauXY = pa.tauXY;

    size_t const Nx = pa.Nx, Ny = pa.NyS;
    const double dX = pa.dX, dY = pa.dY;
    const double dT = pa.dt;
    const double rho = pa.rho0;
    const double dampX = pa.dampX, dampY = pa.dampY;

    // motion equation
    if (i > 0 && i < Nx && j > 0 && j < Ny - 1) {
        Vx[j * (Nx + 1) + i] = Vx[j * (Nx + 1) + i] * (1.0 - dT * dampX) + (dT / rho) * ((
            -P[j * Nx + i] + P[j * Nx + i - 1] + tauXX[j * Nx + i] - tauXX[j * Nx + i - 1]
            ) / dX + (
                tauXY[j * (Nx - 1) + i - 1] - tauXY[(j - 1) * (Nx - 1) + i - 1]
                ) / dY);

        Ux[j * (Nx + 1) + i] = Ux[j * (Nx + 1) + i] + Vx[j * (Nx + 1) + i] * dT;
    }

    if (i > 0 && i < Nx - 1 && j > 0 && j < Ny) {
        Vy[j * Nx + i] = Vy[j * Nx + i] * (1.0 - dT * dampY) + (dT / rho) * ((
            -P[j * Nx + i] + P[(j - 1) * Nx + i] + tauYY[j * Nx + i] - tauYY[(j - 1) * Nx + i]
            ) / dY + (
                tauXY[(j - 1) * (Nx - 1) + i] - tauXY[(j - 1) * (Nx - 1) + i - 1]
                ) / dX);

        Uy[j * Nx + i] = Uy[j * Nx + i] + Vy[j * Nx + i] * dT;
    }
}


__global__ void ComputeStress(kernel_params_t const pa)
{
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const j = blockIdx.y * blockDim.y + threadIdx.y;

    double const* const Ux = pa.Ux;
    double const* const Uy = pa.Uy;
    double* const P = pa.P;
    double* const tauXX = pa.tauXX;
    double* const tauYY = pa.tauYY;
    double* const tauXY = pa.tauXY;

    size_t const Nx = pa.Nx, Ny = pa.NyS;
    const double dX = pa.dX, dY = pa.dY;

    if (i >= Nx || j >= Ny)
        return;

    // constitutive equation - Hooke's law
    double const P0 = pa.P0 * (tex3D<char>(pa.mdata, i + 0.5, j + 0.5, 0.5) == 0);                                       // init pressure

    P[j * Nx + i] = P0 - tex3D<float>(pa.K, i + 0.5, j + 0.5, 0.5) * (
        (Ux[j * (Nx + 1) + i + 1] - Ux[j * (Nx + 1) + i]) / dX + (Uy[(j + 1) * Nx + i] - Uy[j * Nx + i]) / dY            // divU
    );

    tauXX[j * Nx + i] = 2.0 * tex3D<float>(pa.G, i + 0.5, j + 0.5, 0.5) * (
        (Ux[j * (Nx + 1) + i + 1] - Ux[j * (Nx + 1) + i]) / dX -                                                         // dUx/dx -
        ((Ux[j * (Nx + 1) + i + 1] - Ux[j * (Nx + 1) + i]) / dX + (Uy[(j + 1) * Nx + i] - Uy[j * Nx + i]) / dY) / 3.0    // - divU / 3.0
    );

    tauYY[j * Nx + i] = 2.0 * tex3D<float>(pa.G, i + 0.5, j + 0.5, 0.5) * (
        (Uy[(j + 1) * Nx + i] - Uy[j * Nx + i]) / dY -                                                                   // dUy/dy -
        ((Ux[j * (Nx + 1) + i + 1] - Ux[j * (Nx + 1) + i]) / dX + (Uy[(j + 1) * Nx + i] - Uy[j * Nx + i]) / dY) / 3.0    // - divU / 3.0
    );

    if (tex3D<char>(pa.mdata, i + 0.5, j + 0.5, 0.5) == 0)
    {
        P[j * Nx + i] = 0.0;
        tauXX[j * Nx + i] = 0.0;
        tauYY[j * Nx + i] = 0.0;
    }

    if (i < Nx - 1 && j < Ny - 1) {
        tauXY[j * (Nx - 1) + i] = tex3D<float>(pa.G, i + 1.0, j + 1.0, 0.5) * (
            (Ux[(j + 1) * (Nx + 1) + i + 1] - Ux[j * (Nx + 1) + i + 1]) / dY +    // dUx/dy + 
            (Uy[(j + 1) * Nx + i + 1] - Uy[(j + 1) * Nx + i]) / dX                // + dUy/dx
        );

        if (tex3D<char>(pa.mdata, i + 1.0, j + 1.0, 0.5) == 0)
        {
            tauXY[j * (Nx - 1) + i] = 0.0;
        }
    }
}


int main(int argc, char** argv) {

    if (argc < 2) { std::cout << "missing config file\n"; exit(EXIT_FAILURE); }

    std::string config_filename = argv[1];
    std::ifstream config_file(config_filename, std::ios::in);
    if (!config_file) { std::cout << "cannot open config file: " << config_filename << '\n'; exit(EXIT_FAILURE); }

    auto config = nlohmann::json::parse(config_file);

    // input parameters
    size_t const niter   = config["niter"];
    double const eiter   = config["eiter"];
    size_t const Nx      = config["mesh_size"][0];
    size_t const Ny      = config["mesh_size"][1];
    size_t const outstep = config["output_step"];

    // load
    double const load_value = config["load_value"];
    std::array<double, 3> const load_type = config["load_type"];

    std::vector<size_t> devices;
    try
    {
        devices = config["devices"].get<std::vector<size_t>>();
    }
    catch (std::exception const& e)
    {
        devices = std::vector<size_t>({ config["devices"].get<size_t>() });
    }

    // cuda
    dim3 grid, block;
    std::vector<texture_wrapper_t> tex;
    std::vector<kernel_params_t> kernel_pa(devices.size());
    std::vector<cudaStream_t> streams(devices.size());

    // material data
    auto mdata = read_file<char>(config["matrix"], Nx * Ny);
    auto Kdata = read_file<float>(config["K"], Nx * Ny);
    auto Gdata = read_file<float>(config["G"], Nx * Ny);

    for (int device_idx = 0; device_idx < devices.size(); device_idx++)
    {
        gpuErrchk(cudaSetDevice(devices[device_idx]));

        gpuErrchk(cudaStreamCreateWithFlags(&streams[device_idx], cudaStreamNonBlocking));

        // constants
        kernel_pa[device_idx].dt    = config["dt"];
        kernel_pa[device_idx].dX    = config["dx"];
        kernel_pa[device_idx].dY    = config["dy"];
        kernel_pa[device_idx].Lx    = config["phys_size"][0];
        kernel_pa[device_idx].Ly    = config["phys_size"][1];
        kernel_pa[device_idx].dampX = config["dampx"];
        kernel_pa[device_idx].dampY = config["dampy"];
        kernel_pa[device_idx].rho0  = config["rho0"];
        kernel_pa[device_idx].coh   = config["coh"];
        kernel_pa[device_idx].P0    = kernel_pa[device_idx].coh;
        kernel_pa[device_idx].Nx    = Nx;
        kernel_pa[device_idx].Ny    = Ny;

        // slice size & shift
        size_t ysize  = Ny / devices.size() + 2;
        size_t yshift = (ysize - 2) * device_idx - 1;

        if (device_idx == 0)
        {
            ysize -= 1;
            yshift = 0;
        }

        if (device_idx == devices.size() - 1)
            ysize = Ny - yshift;

        kernel_pa[device_idx].NyS    = ysize;
        kernel_pa[device_idx].yshift = yshift;

        // textures
        auto mtex = set_tex(mdata + Nx * yshift, Nx, ysize);
        kernel_pa[device_idx].mdata = mtex.tex_obj;
        tex.emplace_back(mtex);

        auto Ktex = set_tex(Kdata + Nx * yshift, Nx, ysize);
        kernel_pa[device_idx].K = Ktex.tex_obj;
        tex.emplace_back(Ktex);

        auto Gtex = set_tex(Gdata + Nx * yshift, Nx, ysize);
        kernel_pa[device_idx].G = Gtex.tex_obj;
        tex.emplace_back(Gtex);

        // space arrays
        // stress
        set_matrix_zero(&kernel_pa[device_idx].P    , Nx    , ysize    );
        set_matrix_zero(&kernel_pa[device_idx].tauXX, Nx    , ysize    );
        set_matrix_zero(&kernel_pa[device_idx].tauYY, Nx    , ysize    );
        set_matrix_zero(&kernel_pa[device_idx].tauXY, Nx - 1, ysize - 1);

        // displacement
        set_matrix_zero(&kernel_pa[device_idx].Ux, Nx + 1, ysize    );
        set_matrix_zero(&kernel_pa[device_idx].Uy, Nx    , ysize + 1);

        // velocity
        set_matrix_zero(&kernel_pa[device_idx].Vx, Nx + 1, ysize    );
        set_matrix_zero(&kernel_pa[device_idx].Vy, Nx    , ysize + 1);
    }

    delete[] mdata;
    delete[] Kdata;
    delete[] Gdata;

    block.x = BLOCK_SIZE_2D_X;
    block.y = BLOCK_SIZE_2D_Y;
    block.z = 1;
    grid.x = Nx / BLOCK_SIZE_2D_X + 1;
    grid.y = kernel_pa[0].NyS / BLOCK_SIZE_2D_Y + 1;
    grid.z = 1;

    std::array<double, 3> strain = { 
        load_value * load_type[0], 
        load_value * load_type[1], 
        load_value * load_type[2] 
    };

    auto const start = std::chrono::system_clock::now();

    for (int device_idx = 0; device_idx < devices.size(); device_idx++)
    {
        gpuErrchk(cudaSetDevice(devices[device_idx]));
        SetDisp<<<grid, block, 0, streams[device_idx]>>>(strain[0], strain[1], strain[2], kernel_pa[device_idx]);
    }

    double error = 0.0;
    size_t iter = 0;

    for (; iter < niter; iter++) {

        // compute stress
        for (int device_idx = 0; device_idx < devices.size(); device_idx++)
        {
            gpuErrchk(cudaSetDevice(devices[device_idx]));
            ComputeStress<<<grid, block, 0, streams[device_idx]>>>(kernel_pa[device_idx]);
            ComputeDisp<<<grid, block, 0, streams[device_idx]>>>(kernel_pa[device_idx]);
        }

        for (int device_idx = 0; device_idx < devices.size(); device_idx++)
            gpuErrchk(cudaStreamSynchronize(streams[device_idx]));

        // copy displacement between devices before next step
        for (int device_idx = 1; device_idx < devices.size(); device_idx++)
        {
            gpuErrchk(cudaMemcpyPeerAsync(
                kernel_pa[device_idx].Ux,
                devices[device_idx],
                kernel_pa[device_idx - 1].Ux + (Nx + 1) * (kernel_pa[device_idx - 1].NyS - 2),
                devices[device_idx - 1],
                (Nx + 1) * sizeof(double),
                streams[device_idx]
            ));

            gpuErrchk(cudaMemcpyPeerAsync(
                kernel_pa[device_idx].Uy,
                devices[device_idx],
                kernel_pa[device_idx - 1].Uy + Nx * (kernel_pa[device_idx - 1].NyS - 2),
                devices[device_idx - 1],
                Nx * sizeof(double),
                streams[device_idx]
            ));
        }

        for (int device_idx = 0; device_idx < devices.size() - 1; device_idx++)
        {
            gpuErrchk(cudaMemcpyPeerAsync(
                kernel_pa[device_idx].Ux + (Nx + 1) * (kernel_pa[device_idx].NyS - 1),
                devices[device_idx],
                kernel_pa[device_idx + 1].Ux + (Nx + 1),
                devices[device_idx + 1],
                (Nx + 1) * sizeof(double),
                streams[device_idx]
            ));

            gpuErrchk(cudaMemcpyPeerAsync(
                kernel_pa[device_idx].Uy + Nx * kernel_pa[device_idx].NyS,
                devices[device_idx],
                kernel_pa[device_idx + 1].Uy + 2 * Nx,
                devices[device_idx + 1],
                Nx * sizeof(double),
                streams[device_idx]
            ));
        }

        // sync before next step
        for (int device_idx = 0; device_idx < devices.size(); device_idx++)
            gpuErrchk(cudaStreamSynchronize(streams[device_idx]));

        // calc error
        if ((iter + 1) % outstep == 0) {

            double2 vmax = { 0.0, 0.0 };
            for (int device_idx = 0; device_idx < devices.size(); device_idx++)
            {
                gpuErrchk(cudaSetDevice(devices[device_idx]));
                
                vmax.x = std::max(vmax.x, amax(kernel_pa[device_idx].Vx + Nx + 1, (Nx + 1) * (kernel_pa[device_idx].NyS - 1)));
                vmax.y = std::max(vmax.y, amax(kernel_pa[device_idx].Vy + Nx, Nx * kernel_pa[device_idx].NyS));
            }

            error = (vmax.x / kernel_pa[0].Lx + vmax.y / kernel_pa[0].Ly) * kernel_pa[0].dt /
                std::max({ std::abs(strain[0]), std::abs(strain[1]), std::abs(strain[2]) });

            std::cout << "\titeration: " << std::setw(7) << (iter + 1) << ", error = " << std::scientific << error << std::endl;

            if (abs(error) < eiter)
            {
                iter++;
                break;
            }
        }
    }

    auto const end = std::chrono::system_clock::now();

    // bandwidth
    double milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double bandwidth = static_cast<double>((
        2 + // read Ux, Uy in ComputeStress
        4 + // write tauXX, tauYY, tauXY, P in ComputeStress
        6 + // read Vx, Vy, tauXX, tauYY, tauZZ, P in ComputeDisp
        4   // write Ux, Uy, Vx, Vy in ComputeDisp
        ) * iter * sizeof(double) * (Nx + 1) * (Ny + 1)) / milliseconds / 1.0e6;
    std::cout
        << "\n"
        << "iterations : " << iter << "\n"
        << "time       : " << milliseconds << " ms\n"
        << "bandwidth  : " << std::fixed << bandwidth << " GB/s\n"
        << std::endl;

    // write output data
    double* buff = new double[(Nx + 1) * (kernel_pa[0].NyS + 3)];
    for (int device_idx = 0; device_idx < devices.size(); device_idx++)
    {
        gpuErrchk(cudaSetDevice(devices[device_idx]));

        size_t const Ny = kernel_pa[0].NyS;
        save_dev_arr(kernel_pa[device_idx].P, "P_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, Nx * Ny * sizeof(double));
        save_dev_arr(kernel_pa[device_idx].tauXX, "tauXX_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, Nx * Ny * sizeof(double));
        save_dev_arr(kernel_pa[device_idx].tauYY, "tauYY_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, Nx * Ny * sizeof(double));
        save_dev_arr(kernel_pa[device_idx].tauXY, "tauXY_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, (Nx - 1) * (Ny - 1) * sizeof(double));
        save_dev_arr(kernel_pa[device_idx].Ux, "Ux_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, (Nx + 1) * Ny * sizeof(double));
        save_dev_arr(kernel_pa[device_idx].Uy, "Uy_" + std::to_string(Nx) + "_p" + std::to_string(device_idx) + ".dat", buff, Nx * (Ny + 1) * sizeof(double));
    }

    delete[] buff;

    // cleanup
    // materials
    for (auto& t : tex)
    {
        gpuErrchk(cudaSetDevice(t.dev_idx));
        gpuErrchk(cudaDestroyTextureObject(t.tex_obj));
        gpuErrchk(cudaFreeArray(t.arr_obj));
    }

    for (int device_idx = 0; device_idx < devices.size(); device_idx++)
    {
        gpuErrchk(cudaSetDevice(devices[device_idx]));

        // stress
        gpuErrchk(cudaFree(kernel_pa[device_idx].P));
        gpuErrchk(cudaFree(kernel_pa[device_idx].tauXX));
        gpuErrchk(cudaFree(kernel_pa[device_idx].tauYY));
        gpuErrchk(cudaFree(kernel_pa[device_idx].tauXY));

        // displacement
        gpuErrchk(cudaFree(kernel_pa[device_idx].Ux));
        gpuErrchk(cudaFree(kernel_pa[device_idx].Uy));

        // velocity
        gpuErrchk(cudaFree(kernel_pa[device_idx].Vx));
        gpuErrchk(cudaFree(kernel_pa[device_idx].Vy));
    }

    return EXIT_SUCCESS;
}