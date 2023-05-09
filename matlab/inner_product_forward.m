function [output] = inner_product_forward(input, layer, param)

    d = size(input.data, 1);    % height x width x channel
    k = size(input.data, 2);    % batch size
    n = size(param.w, 2);       % # of cols: width
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % logic:    - define a 2d matrix for the output data
    %           - for every batch...
    %               - calculate f(x) = wx + b for every element of x
    %           - define all other output fields
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Replace the following line with your implementation.
    output.data = zeros([n, k]); % 25 x 2, all 0s

    for i = 1:k
        output.data(:, i) = transpose(input.data(:, i)) * param.w + param.b;
    end
    
    output.height = input.height;
    output.width = input.width;
    output.channel = input.channel;
    output.batch_size = input.batch_size;

end