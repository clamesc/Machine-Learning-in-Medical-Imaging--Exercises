% setup MatConvNet
run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn

% Create DAGNN object
net = dagnn.DagNN();

%Create Building Blocks
poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);

%Create Network Layers
net.addLayer('conv3x3_01', convBlock(3,3,3,32), {'input'}, {'x02'}, {'f01', 'b01'});
net.addLayer('relu_01', dagnn.ReLU(), {'x02'}, {'x03'}, {});
net.addLayer('pool_01', poolBlock, {'x03'}, {'x04'}, {});

net.addLayer('conv3x3_02', convBlock(3,3,32,64), {'x04'}, {'x05'}, {'f02', 'b02'});
net.addLayer('relu_02', dagnn.ReLU(), {'x05'}, {'x06'}, {});
net.addLayer('pool_02', poolBlock, {'x06'}, {'x07'}, {});

net.addLayer('conv3x3_03', convBlock(3,3,64,128), {'x07'}, {'x08'}, {'f03', 'b03'});
net.addLayer('relu_03', dagnn.ReLU(), {'x08'}, {'x09'}, {});
net.addLayer('pool_03', poolBlock, {'x09'}, {'x10'}, {});

net.addLayer('conv3x3_04', convBlock(3,3,128,256), {'x10'}, {'x11'}, {'f04', 'b04'});
net.addLayer('relu_04', dagnn.ReLU(), {'x11'}, {'x12'}, {});
net.addLayer('pool_04', poolBlock, {'x12'}, {'x13'}, {});

net.addLayer('fc_05', dagnn.Conv('size', [4 4 256 2048], 'hasBias', true), {'x13'}, {'x14'}, {'f05', 'b05'});
net.addLayer('relu_05', dagnn.ReLU(), {'x14'}, {'x15'}, {});

net.addLayer('fc_06', dagnn.Conv('size', [1 1 2048 2], 'hasBias', true), {'x15'}, {'x16'}, {'f06', 'b06'});
net.addLayer('relu_06', dagnn.ReLU(), {'x16'}, {'prediction'}, {});

%Initialise random parameters
net.initParams();

%Visualize Network
net.print({'input', [64 64 3]}, 'all', true, 'format', 'dot')

%Receptive Fields
net.getVarReceptiveFields('input').size


