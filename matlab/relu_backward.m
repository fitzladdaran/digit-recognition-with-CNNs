function [input_od] = relu_backward(output, input, layer)

    % Replace the following line with your implementation.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % theory:   - relu-forward is defined as follows:
    %               - relu(x) = x, x > 0
    %                         = 0, otherwise
    %           - therefore, the derivative of relu(x) is as follows:
    %               - d(relu(x)) = 1, x > 0
    %                            = 0, otherwise
    %               - note: dh_i/dh_(i-1) = d(relu(x))
    %           - input_od = dl/dh_(i-1) = dl/dh_i * dh_i/dh_(i-1)
    %               - therefore, dl/dh_(i-1) = dl/dh_i, if x > 0
    %                                          0      , otherwise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % logic:    - define input_od to be the same structure as input.data
    %           - define dimensions of input_od
    %           - since input_od = dl/dh_(i-1), define input_od to be of 
    %             equal size as input, filled with 0s
    %           - iterate through input data...
    %               - if the input > 0 (and therefore, d(relu(input)) = 1)
    %                   - dl/dh_(i-1) = dl/dh_i * 1 = dl/dh_i
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    input_od = zeros(size(input.data));
    height = size(input_od, 1);
    width = size(input_od, 2);
    
    for i = 1:height
        for j = 1:width
            if input.data(i, j) > 0 
                input_od(i, j) = output.diff(i, j);
            end
        end
    end

end
