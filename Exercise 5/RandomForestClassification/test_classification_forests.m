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

function [xtrain,ytrain,xtest,ytest] = high_dimensional_data(NbRelevantFeatures)
%% Create high dimensional data - Question b)

N = 2000; % Number of samples of each class (in both training set ans test set)
DimFeatureSpace = 100; % Dimension of the feature space
%NbRelevantFeatures = 20; % How many features actually carry some information 
Relevance = 0.6; % Number between 0.5 and 1. Quantifies how strong the relevant features are. 0.5 : the feature is irrelevant, 1 : the feature is perfectly discriminative
ThresholdType = 'binary'; % In the generated datasets, feature vectors have binary components (0 or 1). Therefore, if a dimension is chosen, a split is automatically defined by the threshold 0.5

[xtrain,ytrain,xtest,ytest] = generateHighDimSets(N,DimFeatureSpace,NbRelevantFeatures,Relevance); % Generate training and test sets
end


function [error_train, error_test] = train_forest(xtrain,ytrain,xtest,ytest, NbTrees, TreeDepth, NbTries)
%% Training parameters

DimFeatureSpace = 2; % Dimension of the feature space: no reason to change this here
ThresholdType = 'random'; % For each dimension, a split with respect to this dimension is defined by a randomly drawn threshold

NbClass = 2; % Number of classes in the label space : don't change this
Bootstrap = 1; % bootstraping (= bagging) coefficient. We do not use bagging in this exercise, so we take 100% of the data set.
NbTrees = 10; % number of trees in the forest
TreeDepth = 10; % maximal depth of the trees
NbTries = 10; % number of tries in each node


%% Create forest structure
obj = ClassificationRandomForests(DimFeatureSpace, NbClass, NbTrees, TreeDepth, Bootstrap, NbTries, ThresholdType);

%% Training
obj = performTraining(obj,xtrain,ytrain);

%% Perform predictions on both training and test sets
Y = computePredictions(obj,xtrain,'average');
[~, pred_train] = max(Y,[],2);
Y = computePredictions(obj,xtest,'average');
[~, pred_test] = max(Y,[],2);


%% Compute train and test errors
error_train = sum(ytrain~=pred_train)/numel(ytrain); % error on the training set
error_test = sum(ytest~=pred_test)/numel(ytest); % error on the test set
end

%% Visualize results (Question a only !)
% . = real class, o = prediction
%figure; clf; hold on; scatter(xtrain(ytrain==1,1),xtrain(ytrain==1,2),'.b'); scatter(xtrain(ytrain==2,1),xtrain(ytrain==2,2),'.r'); scatter(xtrain(pred_train==1,1),xtrain(pred_train==1,2),'bo'); scatter(xtrain(pred_train==2,1),xtrain(pred_train==2,2),'ro');
%figure; clf; hold on; scatter(xtest(ytest==1,1),xtest(ytest==1,2),'.b'); scatter(xtest(ytest==2,1),xtest(ytest==2,2),'.r'); scatter(xtest(pred_test==1,1),xtest(pred_test==1,2),'bo'); scatter(xtest(pred_test==2,1),xtest(pred_test==2,2),'ro');


