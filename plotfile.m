function plotfile(fname, fext, Nx, Ny, plt_title, plt_order, Nparts)
    data = zeros(Ny, Nx);
    pos  = 0;

    for p=0:Nparts-1
        fil = fopen(strcat(fname, int2str(p), fext), 'rb');
        pdata = fread(fil, 'double');
        fclose(fil);
        ysz = length(pdata) / Nx;
        pdata = reshape(pdata, Nx, ysz);
        pdata = transpose(pdata);

        ycp = ysz - 2;
        ysh = 1;
        if p == 0
            ycp = ycp + 1;
            ysh = 0;
        end
        if p == Nparts - 1
            ycp = ycp + 1;
        end

        data(pos + 1:pos + ycp, :) = pdata(ysh + 1:ysh + ycp, :);
        pos = pos + ycp;
    end

    subplot(plt_order(1), plt_order(2), plt_order(3))
    imagesc(data)
    colorbar
    title(plt_title)
    axis image
    set(gca, 'FontSize', 10, 'fontWeight', 'bold')
end