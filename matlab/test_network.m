%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% logic:    - notice that iteration is in stepsizes of 100
%           - notice that P is the probability matrix (as commented
%             in convnet_forward.m), the predictions of our model
%           - since we have P...
%               - for each column of P...
%                   - obtain the max value (highest probability)
%                   - obtain the index of the max value
%                   - save both the max value and the index 
%           - display confusion matrix: y-test vs. predictions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predictions = zeros(1, size(xtest, 2));

for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [probability, index] = max(P);
    predictions(:, i:i+99) = index;
end

confusionchart(confusionmat(ytest, predictions), 0:9)
