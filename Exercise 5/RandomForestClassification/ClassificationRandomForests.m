classdef ClassificationRandomForests
    
    %% CLASS PROPERTIES
    properties
       
        TreeNb = 100;   % number of trees in forests
        TreeDepth = 5;  % number of levels in each tree structure
        InputDim        % Input dimensionality
        NbClasses       % Output dimensionality
        Bootstrap = 1   % percentage of datapoints selected for each bootstrap
        ThresholdType = 'random';
        Ntry = 10;      % nb of tries during optimization
        Trees
        % All the ferns are stored inside
        
        
    end
    
    %% CLASS METHODS
    methods
        
        % constructor
        function obj = ClassificationRandomForests(input_dim, nbClasses, treeNb, treeDepth, bootstrap, nTry, ThresholdType)
            
            % first check attribute
			if(nargin==0)
				return % return emtpy object
			end
            if(nargin==1)
               error('Constructor error: you need to give at least the number of class and the dimensionality');
            end
            if(nargin==2)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
            end
            if(nargin==3)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
               obj.TreeNb = treeNb; 
            end
            if(nargin==4)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
               obj.TreeNb = treeNb; 
               obj.TreeDepth = treeDepth; 
            end
            if(nargin==5)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
               obj.TreeNb = treeNb; 
               obj.TreeDepth = treeDepth;
               obj.Bootstrap = bootstrap;
            end
            if(nargin==6)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
               obj.TreeNb = treeNb; 
               obj.TreeDepth = treeDepth;
               obj.Bootstrap = bootstrap;
               obj.Ntry = nTry;
            end
            if(nargin==7)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;
               obj.TreeNb = treeNb; 
               obj.TreeDepth = treeDepth;
               obj.Bootstrap = bootstrap;
               obj.Ntry = nTry;
               obj.ThresholdType = ThresholdType;
            end
            
            
            % create the ensemble structure
            obj.Trees = cell(obj.TreeNb,1);
            
            for i=1:obj.TreeNb
               
                obj.Trees{i,1} = ClassificationRandomTree(obj.InputDim, obj.NbClasses, obj.TreeDepth, obj.Ntry, obj.ThresholdType);
                
            end
            
        end
        
        % Regression function
        function obj = performTraining(obj,X,Y)
            
            for i=1:obj.TreeNb
                
                if(obj.Bootstrap==1)
                    % we use the full training set
                    obj.Trees{i,1} = performTraining(obj.Trees{i,1},X,Y);
                else
                    % we create a bootstrap of the full training set
                    N = size(X,1);
                    n = round(obj.Bootstrap*N);
                    idx = randi(N,n,1);
                    Xboot = X(idx,:);
                    Yboot = Y(idx,:);
                    obj.Trees{i,1} = performTraining(obj.Trees{i,1},Xboot,Yboot);
                end
            end
            
        end
        

        % Prediction function
        function [Y,Entropy] = computePredictions(obj,X,mode)
            
            if(nargin<3)
                mode = 'average';
            end
            
            if(strcmp(mode,'average'))
                Y = zeros(size(X,1),obj.NbClasses,'single');
                Entropy = zeros(size(X,1),1);
                
                for i = 1:obj.TreeNb
                    
                    %[pred,p] = computePredictions(obj.Trees{i,1},X);
                    [pred,p] = computeFastPredictions(obj.Trees{i,1},X);
                    
                    % combine predictions
                    Y = Y + (1/obj.TreeNb).*pred;
                    Entropy = Entropy + (1/obj.TreeNb).*p;
                    
                end
            elseif(strcmp(mode,'bestpred'))
                Y = zeros(size(X,1),obj.NbClasses,'single');
                Entropy = Inf.*ones(size(X,1),1);
                
                for i = 1:obj.TreeNb
                    
                    %[pred,p] = computePredictions(obj.Trees{i,1},X);
                    [pred,p] = computeFastPredictions(obj.Trees{i,1},X);
                    idx = p<Entropy;
                    
                    % combine predictions
                    Y(idx,:) = pred(idx,:);
                    Entropy(idx,:) = p(idx,:);
                    
                end
                
            elseif(strcmp(mode,'allpred'))
                Y = zeros(size(X,1),obj.NbClasses,obj.TreeNb,'single');
                Entropy = zeros(size(X,1),obj.TreeNb);
                
                for i = 1:obj.TreeNb
                    
                    %[pred,p] = computePredictions(obj.Trees{i,1},X);
                    [pred,p] = computeFastPredictions(obj.Trees{i,1},X);
                    
                    % keep all predictions
                    Y(:,:,i) = pred;
                    Entropy(:,i) = p;
                    
                end
                
                
            end
            
            
        end
        
        function saveobj(obj, str)
		
			% first create saving directory
			mkdir(str);
			
			% then go into the directory
			
			cd(str)
			
			% save forest attributes
			TreeNb = obj.TreeNb;
			TreeDepth = obj.TreeDepth; 
			InputDim = obj.InputDim;
			NbClasses = obj.NbClasses;
			Bootstrap = obj.Bootstrap;
			Ntry = obj.Ntry;
			
			save('RF.mat', 'TreeNb', 'TreeDepth', 'InputDim', 'NbClasses', 'Bootstrap', 'Ntry');
			
			% then save all trees
			for i=1:TreeNb
			
				saveobj(obj.Trees{i,1},num2str(i));
			
			end
			
			% go back
			cd ..
		
		end
		
		function obj = loadobj(obj,str)
		
			% need as input an empty object
		
			if(nargin==2)
				cd(str)
			end
			
			% first look for the RF.mat file and load forest attributes
			load('RF.mat')
			obj = ClassificationRandomForests(InputDim, NbClasses, TreeNb, TreeDepth, Bootstrap, Ntry);
			
			% then load the trees
			for i=1:TreeNb
                tree = ClassificationRandomTree();
				obj.Trees{i,1} = loadobj(tree, num2str(i));
			end
			
			if(nargin==2)
				cd ..
			end
		
		end
        
    end
    
end