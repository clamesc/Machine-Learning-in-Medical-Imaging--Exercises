classdef ClassificationRandomTree
    
    
    %% CLASS PROPERTIES
    properties 
       
        TreeDepth = 5;       % number of levels in tree structure
        PopulationPerLeaf = 1;
        ThresholdType = 'random';
        InputDim             % dimensionality of data
        NbClasses            % dimensionality of predictions
        Labels               % index of the different classes 
        Ntry = 10;
        Nodes                % random nodes - corresponds to the chosen dimensionality for best split
        Thr                  % random thresholds
        Posteriors           % probability distribution for each class
        Priors               % initial class cardinals
        Entropy           % entropy measure

        
    end
    
    %% CLASS METHODS
    methods
        
        
        % constructor
        function obj = ClassificationRandomTree(input_dim, nbClasses, treeDepth, ntry, ThresholdType)
            
             % first check attribute
			if(nargin==0)
				return; % return empty object
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
               obj.TreeDepth = treeDepth; 
            end
            if(nargin==4)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;  
               obj.TreeDepth = treeDepth; 
               obj.Ntry = ntry;
            end
            if(nargin==5)
               obj.InputDim = input_dim; 
               obj.NbClasses = nbClasses;  
               obj.TreeDepth = treeDepth; 
               obj.Ntry = ntry;
               obj.ThresholdType = ThresholdType;
            end
           
            
        end
        
        % Regression function - tree growing
        function obj = performTraining(obj,X,Y)
            
            treeDepth = obj.TreeDepth;
            
            % check cardinality of each class
            obj.Labels = unique(Y);
            priors = zeros(1,obj.NbClasses);
            for i=1:obj.NbClasses
               priors(1,i) = sum(Y==obj.Labels(i)); 
            end
            obj.Priors = priors;
            
            % perform recursive splitting
            nodes_count = 0;
            parent = 0;
            obj = performSplitting(obj,X,Y,treeDepth,nodes_count,parent);
            
        end
        
        % recursive splitting
        function obj = performSplitting(obj,X,Y,treeDepth,nodes_count,parent,leftright)
            
            % first increment the nodes 
            nodes_count = nodes_count + 1;
            
            % ------ Compute Node statistics and initialize split func -------- %
            obj.Nodes{nodes_count,1} = -1;
            obj.Nodes{nodes_count,2} = -1;  %left child
            obj.Nodes{nodes_count,3} = -1; % right child
            obj.Thr{nodes_count,1} = -1;
            obj.Thr{nodes_count,2} = -1;  %left child
            obj.Thr{nodes_count,3} = -1; % right child
            
            posts = hist(Y,obj.Labels);
            posts = posts./obj.Priors;
            posts = posts./sum(posts);
            
            obj.Posteriors{nodes_count,1} = posts;
            obj.Posteriors{nodes_count,2} = -1;  %left child
            obj.Posteriors{nodes_count,3} = -1; % right child
            
            entropy = sum(-posts(posts>0).*(log(posts(posts>0))));
            obj.Entropy{nodes_count,1} = entropy;
            obj.Entropy{nodes_count,2} = -1;  %left child
            obj.Entropy{nodes_count,3} = -1; % right child
           
            
            % set up link with parent
            if(parent>0)
               
               switch leftright    % 1 = left and 2 = right  
                
                   case 1 
                       obj.Nodes{parent,2} = nodes_count;
                       obj.Thr{parent,2} = nodes_count;
                       obj.Posteriors{parent,2} = nodes_count;
                       obj.Entropy{parent,2} = nodes_count;
                   
                   case 2
                       obj.Nodes{parent,3} = nodes_count;
                       obj.Thr{parent,3} = nodes_count;
                       obj.Posteriors{parent,3} = nodes_count;
                       obj.Entropy{parent,3} = nodes_count;
                
               end
               
            end
            
            % FIRTS CHECK if we have only one class, then we stop
            if(numel(unique(Y))<2 )
                                
                return;
                
            end
            
            % ------ Now, we will try to perform the splitting -------- %
            % Here you have 2 solutions, either thresholds are generated
            % randomly, or according to the median
            % first check if max depth has been reached, if not try to split
            if(treeDepth>0)
            
                % check number of points
                NbPoints = size(X,1);
                
                % first we select randomly Ntry dimensions
                D = obj.InputDim;
                
                if ((strcmp(obj.ThresholdType,'median')) || (strcmp(obj.ThresholdType,'binary'))) 
                    dim_idx = randperm(D);
                    dim_idx = dim_idx(1:obj.Ntry);
                end
                
                if (strcmp(obj.ThresholdType,'random'))
                    dim_idx = randi(D,obj.Ntry,1);
                end
                
                % we keep only these dimensions
                
                xx = X(:,dim_idx);
                
                if (strcmp(obj.ThresholdType,'median'))
                    thr = median(xx,1);
                end
                
                if (strcmp(obj.ThresholdType,'binary'))
                    thr = 0.5*ones(1,obj.Ntry);
                end
                
                if (strcmp(obj.ThresholdType,'random'))
                    Xmin = min(xx,[],1);
                    Xmax = max(xx,[],1);
                    thr = Xmin + (Xmax-Xmin).*rand(1,obj.Ntry);
                end
                
                % loop over the candidates
                IG = 0;
                
                for i=1:obj.Ntry
                    
                    x = xx(:,i);
                    
                    % compute left and right indexes
                    
                    idxLeft = x<repmat(thr(i),size(x,1),1);
                    idxRight = x>=repmat(thr(i),size(x,1),1);
                    
                    yLeft = Y(idxLeft,:);
                    yRight = Y(idxRight,:);
                    
                    % check size of left and right
                    NbPointsL = size(yLeft,1);
                    NbPointsR = size(yRight,1);
                    
                    % if both contains enough points then compute info gain
                    if(NbPointsL>obj.PopulationPerLeaf && NbPointsR>obj.PopulationPerLeaf)
                        
                        % compute information gain
                        wl = NbPointsL/NbPoints;
                        wr = NbPointsR/NbPoints;
                        
                        % compute information gain
                        Gamma = obj.Entropy{nodes_count,1};
                        
                        postsL = hist(yLeft,obj.Labels);
                        postsL = postsL./obj.Priors;
                        postsL = postsL./sum(postsL);
                        Gammal = sum(-postsL(postsL>0).*log(postsL(postsL>0)));
                        
                        postsR = hist(yRight,obj.Labels);
                        postsR = postsR./obj.Priors;
                        postsR = postsR./sum(postsR);
                        Gammar = sum(-postsR(postsR>0).*log(postsR(postsR>0)));
                        
                        infoGain = Gamma-wl*Gammal-wr*Gammar;
                        
                        
                        if(infoGain>IG)
                            
                            IG = infoGain;
                            XL = X(idxLeft,:);
                            XR = X(idxRight,:);
                            YL = Y(idxLeft,:);
                            YR = Y(idxRight,:);
                            nodes = dim_idx(i);
                            thresholds = thr(i);
                            
                        end
                    end
                    
                end
                
                
                % now if we have a good split, we can fill the tree nodes and
                % thresholds, and iterate splitting
                
                if(IG==0)
                    
                    return;
                    
                else
                    
                    obj.Nodes{nodes_count,1} = nodes;
                    obj.Thr{nodes_count,1} = double(thresholds);
                    
                    % Perform iteration
                    
                    treeDepth = treeDepth - 1;
                    
                    parent = nodes_count;
                    obj = performSplitting(obj,XL,YL,treeDepth,nodes_count,parent,1);
                    
                    nodes_count = size(obj.Nodes,1);
                    obj = performSplitting(obj,XR,YR,treeDepth,nodes_count,parent,2);
                end
                
            end
        end
        
        % function for computing prediction calling MEX function
        function [Y,E] = computeFastPredictions(obj,X)
            
            nodes = cell2mat(obj.Nodes);
            thr = cell2mat(obj.Thr);
            posts = cell2mat(obj.Posteriors);
            entropy = cell2mat(obj.Entropy);
            [Y,E] = predictMEX(single(X),nodes,thr,posts,entropy);
            
        end
        
        % function for computing predictions
        function [Y,E] = computePredictions(obj,X)
            
            N = size(X,1);
            
            % allocate memory for predictions
            Y = zeros(N,obj.NbClasses);
            E = zeros(N,1);
            
            % loop over the points
            for i=1:N
               
               % initialize tree indexes 
               index = 1;
               %link=0; 
                
               while (index >= 0)
                   
                   % go to the node
                   
                   node = obj.Nodes{index,1};
                   threshold = obj.Thr{index,1};
                   Y(i,:) = obj.Posteriors{index,1:obj.NbClasses};
                   E(i,1) = obj.Entropy{index,1};
                   
				   if(node>-1)
				   
						% perform test
                   
						if(X(i,node)<threshold)
							index = obj.Nodes{index,2}; % then we go left
						else
							index = obj.Nodes{index,3}; % or we go right
						end
				   
				   else
					   index = -1;
				   end
                   
                    
               end
               
                
                
            end
            
            
        end
		
		function saveobj(obj, str)
		
			treename = ['tree_' str '.mat'];
		
			% save tree attributes
			
			TreeDepth = obj.TreeDepth; 
			PopulationPerLeaf = obj.PopulationPerLeaf;
			InputDim = obj.InputDim ;            
			NbClasses = obj.NbClasses ;           
			Labels = obj.Labels ;              
			Ntry = obj.Ntry ;
			Nodes = obj.Nodes ;
			Thr = obj.Thr ; 
			Posteriors  = obj.Posteriors ;         
			Priors = obj.Priors ;              
			Entropy = obj.Entropy ;          
			
			save(treename, 'TreeDepth', 'PopulationPerLeaf', 'InputDim', 'NbClasses', 'Labels', 'Ntry', 'Nodes', 'Thr', 'Posteriors', 'Priors', 'Entropy');
			
		end
		
		function obj = loadobj(obj,str)
			
			treename = ['tree_' str '.mat'];
			
			% load tree attributes
			load(treename);
			
			obj.TreeDepth = TreeDepth; 
			obj.PopulationPerLeaf = PopulationPerLeaf;
			obj.InputDim = InputDim ;            
			obj.NbClasses = NbClasses ;           
			obj.Labels = Labels ;              
			obj.Ntry = Ntry ;
			obj.Nodes = Nodes ;
			obj.Thr = Thr ; 
			obj.Posteriors  = Posteriors ;         
			obj.Priors = Priors ;              
			obj.Entropy = Entropy ;   
		
		end
        
    
    end
    
    
end