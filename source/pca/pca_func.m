function [fc_pca, fc_k, pool_pca, pool_k] = pca_func(fc_vec, pool_vec, ftr_dir, cls_name)

    %%================================================================
    %% POOL 5
    len = 7;
    channel = 512;
    f_num = len * len;

    pool_x = zeros(channel, f_num, 'single');
    pool_data = reshape(cell2mat(pool_vec), [f_num, 512]);
    for id = 1:f_num
        pool_x(:, id) = pool_data(id, :);
    end

    k = len * channel;
    d = pool_x;

    file_name = sprintf('%s\\%s\\pca_base_pool5.bin', ftr_dir, cls_name);
    pool_base_fileID = fopen(file_name, 'r');
    if pool_base_fileID == -1
        err = sprintf('Error: failed reading %s', file_name);
        disp(err);
    else
        k = fread(pool_base_fileID, 1, 'int');
        data = fread(pool_base_fileID, k * channel * 1, 'float32');
        pool_base = reshape(data, [channel, k]);
        pool_base = pool_base';
        fclose(pool_base_fileID);

        x_mean = mean(pool_x, 1);
        x_size = size(pool_x, 1);
        x_remat = repmat(x_mean, x_size, 1);
        pool_x = pool_x - x_remat;
        ux = pool_base * pool_x;

        d = zeros(f_num, k, 'single');
        for id = 1:f_num
            v = ux(:, id);
            d(id, :) = v / norm(v);
        end
    end
    
    pool_k = k;
    pool_pca = reshape(d, [k * f_num, 1]);   
            
    %%=====================================================================
    %% fc6
    channel = 4096;
    f_num = 1;
    
    fc6_x = zeros(channel, f_num, 'single');
    fc6_data = reshape(cell2mat(fc_vec), [f_num, channel]);
    for id = 1:f_num
        fc6_x(:, id) = fc6_data(id, :);
    end
    
    k = channel;
    d = fc6_x;
    
    file_name = sprintf('%s\\%s\\pca_base_fc6.bin', ftr_dir, cls_name);
    fc6_base_fileID = fopen(file_name, 'r'); 

    if fc6_base_fileID == -1
        err = sprintf('Error: failed reading %s', file_name);
        disp(err);
    else
        k = fread(fc6_base_fileID, 1, 'int');
        data = fread(fc6_base_fileID, k * channel * 1, 'float32');
        fc6_base = reshape(data, [channel, k]);
        fc6_base = fc6_base';
        fclose(fc6_base_fileID);

        fc6_x = fc6_x - repmat(mean(fc6_x, 1), size(fc6_x, 1), 1);
        ux = fc6_base * fc6_x;

        d = zeros(f_num, k, 'single');
        for id = 1:f_num
            v = ux(:, id);
            d(id, :) = v / norm(v);
        end
    end
    
    fc_k = k;
    fc_pca = reshape(d, [k * f_num, 1]);
    
end
