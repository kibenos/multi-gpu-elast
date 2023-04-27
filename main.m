clear
figure(1)
clf
colormap jet

loadValue = -0.000015;
loadType = [4.0, -2.0, 0.0];
nGrid = 32;
nIter = 100000;
eIter = 1.0e-10;
devices = [0, 1];
outputStep = 10000;

CFL = 0.5;                                                 % Courant-Friedrichs-Lewy
Nx = 32 * nGrid;                                           % number of space steps
Ny = 32 * nGrid;
Lx = 20.0;                                                 % physical length
Ly = 20.0;                                                 % physical width
rho0 = 1.0;                                                % density
K0 = 1.0;                                                  % bulk modulus
G0 = 0.01;                                                 % shear modulus
N = 1;
coh  = 0.00001 * sqrt(2);
porosity = 0.005;
rad = sqrt(porosity * Lx * Lx / pi / N / N);
dX = Lx / (Nx - 1);                                        % space step
dY = Ly / (Ny - 1);
x = (-Lx / 2) : dX : (Lx / 2);                             % space discretization
y = (-Ly / 2) : dY : (Ly / 2);
[x, y] = ndgrid(x, y);                                     % 2D mesh
dt = CFL * min(dX, dY) / sqrt( (K0 + 4*G0/3) / rho0);      % time step
dampX = 2.0 / dt / Nx;
dampY = 2.0 / dt / Ny;

config = struct();

config.mesh_size   = [ Nx, Ny ];
config.phys_size   = [ Lx, Ly ];
config.matrix      = "m.dat";
config.K           = "K.dat";
config.G           = "G.dat";
config.dx          = dX;
config.dy          = dY;
config.dt          = dt;
config.dampx       = dampX;
config.dampy       = dampX;
config.rho0        = rho0;
config.coh         = coh;
config.load_value  = loadValue;
config.load_type   = loadType;
config.niter       = nIter;
config.eiter       = eIter;
config.devices     = devices;
config.output_step = outputStep;

configJson = jsonencode(config, 'PrettyPrint', true);
fid = fopen('config.json', 'w');
fprintf(fid, '%s', configJson);
fclose(fid);

M = ones(Nx, Ny);
for i = 0 : N - 1
    for j = 0 : N - 1
        M(sqrt((x - 0.5*Lx*(1-1/N)  + (Lx/N)*i) .* (x - 0.5*Lx*(1-1/N) + (Lx/N)*i) + (y - 0.5*Ly*(1-1/N) + (Ly/N)*j) .* (y - 0.5*Ly*(1-1/N) + (Ly/N)*j)) < rad) = 0;
    end
end
K = K0 * M + 0.01 * K0 * ~M;
G = G0 * M + 0.01 * G0 * ~M;

fil = fopen('m.dat', 'wb');
fwrite(fil, M(:), 'int8');
fclose(fil);

fil = fopen('K.dat', 'wb');
fwrite(fil, K(:), 'float');
fclose(fil);

fil = fopen('G.dat', 'wb');
fwrite(fil, G(:), 'float');
fclose(fil);

% GPU CALCULATION
system('nvcc -O 3 main.cu');
system('.\a.exe config.json');

rows = 2;
cols = 3;
parts = length(devices);
plotfile(strcat('P_'    , int2str(Nx), '_p'), '.dat', Nx    , Ny    , 'P'    , [rows, cols, 1], parts);
plotfile(strcat('tauXX_', int2str(Nx), '_p'), '.dat', Nx    , Ny    , 'tauXX', [rows, cols, 2], parts);
plotfile(strcat('tauYY_', int2str(Nx), '_p'), '.dat', Nx    , Ny    , 'tauYY', [rows, cols, 3], parts);
plotfile(strcat('tauXY_', int2str(Nx), '_p'), '.dat', Nx - 1, Ny - 1, 'tauXY', [rows, cols, 4], parts);
plotfile(strcat('Ux_'   , int2str(Nx), '_p'), '.dat', Nx + 1, Ny    , 'Ux'   , [rows, cols, 5], parts);
plotfile(strcat('Uy_'   , int2str(Nx), '_p'), '.dat', Nx    , Ny + 1, 'Uy'   , [rows, cols, 6], parts);
