function [xtrain,ytrain,xtest,ytest] = high_dimensional_data(NbRelevantFeatures)
%% Create high dimensional data - Question b)

N = 2000; % Number of samples of each class (in both training set ans test set)
DimFeatureSpace = 100; % Dimension of the feature space
%NbRelevantFeatures = 20; % How many features actually carry some information 
Relevance = 0.6; % Number between 0.5 and 1. Quantifies how strong the relevant features are. 0.5 : the feature is irrelevant, 1 : the feature is perfectly discriminative
ThresholdType = 'binary'; % In the generated datasets, feature vectors have binary components (0 or 1). Therefore, if a dimension is chosen, a split is automatically defined by the threshold 0.5

[xtrain,ytrain,xtest,ytest] = generateHighDimSets(N,DimFeatureSpace,NbRelevantFeatures,Relevance); % Generate training and test sets
end