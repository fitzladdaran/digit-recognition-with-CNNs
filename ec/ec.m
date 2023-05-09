%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% logic:    - define network
%           - load trained weights
%           - define a variable to hold data of image examples
%           - for every image...
%               - load image into memory
%               - convert to grayscale
%               - calculate a threshold level
%               - binarize the image with the threshold level
%               - flip pixel values 
%                   - black --> white, white --> black
%               - remove isolated pixels
%               - perform morphological closing on image
%               - obtain connected components
%               - obtain statistics of connected components: bounding box
%               - for every bounding box
%                   - take the image within the bounding box
%               - pad image as necessary
%               - resize image to input dimensions: 28 x 28
%               - transpose image
%               - reshape image to 1d vector, then store data
%           - set batch_size to be the entire set of images: 75
%           - call convnet_forward(), similiar to train_network.m
%           - obtain predictions
%           - confusion matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../matlab');

% network definition
layers = get_lenet();

% load trained weights
load lenet.mat

images = zeros(784, 75);                    % set dimensions of images
image_number = 1;                           % iterator 

% for every image input...
for i = 1:4
    image = sprintf('../images/image%d.jpg', i);
    image = rgb2gray(imread(image));
    level = graythresh(image);
    binary_image = imbinarize(image, level);
    binary_image_flipped = ~binary_image;
    binary_image_flipped_clean = bwareaopen(binary_image_flipped, 10, 8);

    % for the fourth image, morph nearby connected components
    if i == 4
        se = strel('disk', 2);
        binary_image_flipped_clean = imclose(binary_image_flipped_clean, se);
    end
    
    % get connected components and bounding box(es)
    connected_components = bwconncomp(binary_image_flipped_clean);
    stats = regionprops(connected_components, 'BoundingBox');
    
    % show image of bounding boxes on binarized images
    %figure
    %imshow(binary_image_flipped_clean);
    %hold on

    % for every bounding box
    for j = 1:length(stats)
        box = stats(j).BoundingBox;
        %rectangle('Position', [box(1), box(2), box(3), box(4)], ...
        %          'EdgeColor', 'y', 'LineWidth', 2);

        % obtain image within bounding box
        number = binary_image_flipped_clean(floor(box(2)): ...
                                            ceil(box(2) + box(4) - 1), ...
                                            floor(box(1)): ...
                                            ceil(box(1) + box(3) - 1));
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % reference: discussed with colleagues on how to pad image;
        %            accordingly, it's based on logical math
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % image padding (obtained from reference)
        bigger_dim = max(size(number, 1), size(number, 2));
        smaller_dim = min(size(number, 1), size(number, 2));
        difference = bigger_dim - smaller_dim;

        if bigger_dim / smaller_dim < 2
            number = padarray(number, [10, 10], 0);
        else
            number = padarray(number, [10, floor(difference / 2) + 1], 0);
        end

        number = imresize(number, [28, 28], 'box'); 
        number = transpose(number);
        images(:, image_number) = reshape(number, 784, 1);
        image_number = image_number + 1;
    end
    
    %hold off
end

test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, ...
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, ...
               6, 0, 6, 2, 6, ...
               7, 0, 9, 3, 1, 6, 7, 2, 6, 1, 3, 9, 6, 4, 1, 4, 2, 0, ...
               0, 5, 4, 4, 7, 3, 1, 0, 2, 5, 5, 1, 7, 7, 4, 9, 1, 7, ...
               4, 2, 9, 1, 5, 3, 4, 0, 2, 9, 4, 4, 1, 1];

% run examples through network
layers{1, 1}.batch_size = 75;
[output, P] = convnet_forward(params, layers, images);
[probability, index] = max(P);

% match index with actual number from 0 to 9
index = index - 1;

% confusion matrix
figure
confusionchart(confusionmat(test_values, index), 0:9)


