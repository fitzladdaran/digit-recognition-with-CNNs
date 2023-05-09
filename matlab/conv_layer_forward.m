function [output] = conv_layer_forward(input, layer, param)
    % Conv layer forward
    % input: struct with input data
    % layer: convolution layer struct
    % param: weights for the convolution layer
    
    % output: 
    
    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    num = layer.num;
    
    % resolve output shape
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    assert(h_out == floor(h_out), 'h_out is not integer')
    assert(w_out == floor(w_out), 'w_out is not integer')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % logic:    - define a 2d matrix for the output data
    %           - create a temp variable to copy input data, to be used as
    %             function parameter in im2col_conv()
    %           - for every batch...
    %               - set temp data to be current batch's data (1d vector)
    %               - convert (temp) image data into column representation 
    %                 of output data dimensions
    %               - reshape image data into appropriate dimensions to 
    %                 calculate f(X, W, b) = X * W + b
    %               - calculate f(X, W, b) = X * W + b
    %               - reshape f into appropriate dimensions, and save to 
    %                 output data
    %           - define all other output fields
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reference: discussed with colleagues
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Fill in the code
    % Iterate over the each image in the batch, compute response,
    % Fill in the output datastructure with data, and the shape. 
    output.data = zeros([h_out * w_out * num, batch_size]);
    temp = input;
    
    for i = 1:batch_size
        temp.data = input.data(:, i);
        col = im2col_conv(temp, layer, h_out, w_out);
        res = reshape(col, k * k * c, h_out * w_out);
        f = transpose(res) * param.w + param.b;
        output.data(:, i) = reshape(f, h_out * w_out * num, 1);
    end
    
    output.height = h_out;
    output.width = w_out;
    output.channel = num;
    output.batch_size = batch_size;
    
end
