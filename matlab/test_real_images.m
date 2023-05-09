%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% logic:    - define network
%           - load trained weights
%           - define a variable to hold data of real-life image examples
%           - for every real-life image example...
%               - load image into memory
%               - convert to grayscale
%               - resize image to input dimensions: 28 x 28
%               - transpose image
%               - reshape image to 1d vector, then store data
%           - set batch_size to be the entire set of images: 5
%           - call convnet_forward(), similiar to train_network.m
%           - obtain predictions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% network definition
layers = get_lenet();

% load trained weights
load lenet.mat

images = zeros(784, 15);

for i = 1:15
    image = sprintf('../images/ex%d.png', i);
    image = rgb2gray(imread(image));
    image = imresize(image, [28 28]);
    image = transpose(image);
    images(:, i) = reshape(image, 784, 1);
end

layers{1, 1}.batch_size = size(images, 2);
[output, P] = convnet_forward(params, layers, images);
[probability, index] = max(P);
index = index - 1;