function [xtrain, ytrain, xtest, ytest] = gaussian_data_2d()

%% Create 2D Gaussian data - Question a)

N = 2000; % Number of samples of each class (in both training set ans test set)
DimFeatureSpace = 2; % Dimension of the feature space: no reason to change this here
ThresholdType = 'random'; % For each dimension, a split with respect to this dimension is defined by a randomly drawn threshold
    
% Training set
xtrain = [randn(N,2); 3+randn(N,2)]; % features
ytrain = [ones(N,1); 2.*ones(N,1)]; % labels
    
% Test set
xtest = [randn(N,2); 3+randn(N,2)];
ytest = [ones(N,1); 2.*ones(N,1)];
end