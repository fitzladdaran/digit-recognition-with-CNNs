function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out * w_out * c, batch_size]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % logic:    - define a 2d matrix for the output data
    %           - create a temp variable to store 3d representation (image)
    %             of 1d input data (vector)
    %           - for every batch...
    %               - reshape 1d vector to 3d image
    %               - pad 3d image with 0s
    %               - iterate through padded image by height by stride
    %                 steps
    %                   - iterate through padded image by width by stride 
    %                     steps
    %                       - iterate through padded image by channel
    %                           - find max value in filter
    %                           - save max value in temp variable
    %           - reshape temp variable into 1d vector
    %           - save 1d vector in output data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    temp = zeros([h_out, w_out, c]);

    for i = 1:batch_size

        % 3d representation of original image (per batch)
        image = reshape(input.data(:, i), h_in, w_in, c);

        % padded original image
        image = padarray(image, [pad, pad], 0);
        padded_h = size(image, 1);                  % padded image: height
        padded_w = size(image, 2);                  % padded image: width
        
        h = 1;                                      % iterator for 'height'

        % iterate through new 'height' bounds by 'stride' stepsize...
        for height = 1:stride:padded_h
            w = 1;                                  % iterator for 'width'

            % iterate through new 'width' bounds by 'stride' stepsize
            for width = 1:stride:padded_w

                % iterate through channels...
                for channel = 1:c
                    temp(h, w, channel) = max(image(height:height + k - 1, ...
                                                    width:width + k - 1, ...
                                                    channel), [], 'all');
                end

                w = w + 1;
            end

            h = h + 1;
        end
        
        output.data(:, i) = reshape(temp, h_out * w_out * c, 1);
    end
    
end
