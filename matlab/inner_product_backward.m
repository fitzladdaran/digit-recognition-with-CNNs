function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

    % Replace the following lines with your implementation.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % theory:   - inner-product-forward is defined as follows:
    %               - IP(x) = wx + b
    %           - dl/dw_i = dl/dh_i * dh_i/dw_i 
    %               - equivalently, output.diff * x
    %           - dl/db = dl/dh_i * dh_i/db_i
    %               - equivalently, output.diff * 1
    %           - dl/dh_(i-1) = dl/dh_i * dh_i/dh_(i-1)
    %               - equivalently, output.diff * w
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reference: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    % reference: https://www.youtube.com/watch?v=tIeHLnjs5U8
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % based on references, dh_i/dw_i = x;
    % note: modified structure to work with code
    param_grad.w = transpose(output.diff * transpose(input.data));

    % based on references, dh_i/db_i = 1;
    % note: modified structure to work with code
    param_grad.b = transpose(sum(output.diff, 2));

    % based on references, dh_i/dh_(i-1) = w;
    % note: modified structure to work with code
    input_od = param.w * output.diff;
    
end
