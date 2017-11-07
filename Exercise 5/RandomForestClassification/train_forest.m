function [error_train, error_test] = train_forest(xtrain,ytrain,xtest,ytest, NbTrees, TreeDepth, NbTries, DimFeatureSpace, ThresholdType)
%% Training parameters

%DimFeatureSpace = 2; % Dimension of the feature space: no reason to change this here
%ThresholdType = 'random'; % For each dimension, a split with respect to this dimension is defined by a randomly drawn threshold

NbClass = 2; % Number of classes in the label space : don't change this
Bootstrap = 1; % bootstraping (= bagging) coefficient. We do not use bagging in this exercise, so we take 100% of the data set.
%NbTrees = 10; % number of trees in the forest
%TreeDepth = 10; % maximal depth of the trees
%NbTries = 10; % number of tries in each node


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