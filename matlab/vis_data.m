layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 

layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);

% Fill in your code here to plot the features.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% logic:    - define (figure) output dimensions
%           - define layers
%           - track channel index
%           - convolution layer:
%               - for every index of the (figure) output...
%                   - create a subplot
%                   - show the image on the subplot
%                   - increment channel index
%           - reset channel index
%           - relu layer:
%               - for every index of the (figure) output...
%                   - create a subplot
%                   - show the image on the subplot
%                   - increment channel index      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define output dimensions
rows = 4;
cols = 5;

% define layers
output_2 = reshape(output{1, 2}.data, output{1, 2}.height, ...
                   output{1, 2}.width, output{1, 2}.channel);
output_3 = reshape(output{1, 3}.data, output{1, 3}.height, ...
                   output{1, 3}.width, output{1, 3}.channel);

% convolution layer
figure('Name', 'Convolution Layer');
channel = 1;
for i = 1:rows
    for j = 1:cols
        subplot(rows, cols, channel);
        imshow(transpose(output_2(:, :, channel)));
        channel = channel + 1;
    end
end

% relu layer
figure('Name', 'RELU Layer');
channel = 1;
for i = 1:rows
    for j = 1:cols
        subplot(rows, cols, channel);
        imshow(transpose(output_3(:, :, channel)));
        channel = channel + 1;
    end
end
