% setup MatConvNet
run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn

% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-resnet-50-dag.mat')) ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('trumpet.jpg') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
net.eval({'data', im_}) ;

net.print({'data', [224 224 3]}, 'all', true)

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

[top, index] = sort(scores(:),'descend');
prob = [top(1);top(2);top(3);top(4);top(5)];
table(prob,'RowNames',{net.meta.classes.description{index(1)};net.meta.classes.description{index(2)};net.meta.classes.description{index(3)};net.meta.classes.description{index(4)};net.meta.classes.description{index(5)}})

% show the classification results
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;