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

%net.print({'data', [224 224 3]}, 'all', true)

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

[top, index] = sort(scores(:),'descend');
prob = [top(1);top(2);top(3);top(4);top(5)];
table(prob,'RowNames',{net.meta.classes.description{index(1)};net.meta.classes.description{index(2)};net.meta.classes.description{index(3)};net.meta.classes.description{index(4)};net.meta.classes.description{index(5)}})

class = net.meta.classes.description{index(1)};
classIdx = index(1);
classProb = top(1);

occlusions = zeros(16,16);
for i = 0:15
    for j = 0:15
        imOcc = im_;
        occlusion = ones(14,14,3);
        rgbMean = mean(mean(im));
        occlusion(:,:,1) = rgbMean(:,:,1);
        occlusion(:,:,2) = rgbMean(:,:,2);
        occlusion(:,:,3) = rgbMean(:,:,3);
        imOcc(i*14+1:i*14+14,j*14+1:j*14+14,:) = occlusion;
        net.eval({'data', imOcc}) ;
        scoresOcc = net.vars(net.getVarIndex('prob')).value ;
        scoresOcc = squeeze(gather(scoresOcc)) ;
        occlusions(i+1,j+1) = classProb - scoresOcc(classIdx);
    end
end

occlusions = occlusions - min(min(occlusions));
occlusions = 255*abs(occlusions)/max(max(abs(occlusions)));
occlusions = ind2rgb(int16(imresize(occlusions,14,'nearest')),colormap(jet(255)));

% show the classification results
[bestScore, best] = max(scores) ;
C = imfuse(im,occlusions,'blend');
figure(1) ; clf ; imshow(C) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;