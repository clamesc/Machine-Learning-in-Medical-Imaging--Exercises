function [Xtrain,Ytrain,Xtest,Ytest] = generateHighDimSets(nbSamplesPerClass,nbFeatures,nbRealFeatures,relevance)

nbDummyFeatures = nbFeatures - nbRealFeatures;
n1 = nbSamplesPerClass;
n2 = nbSamplesPerClass;
nbPoints = n1 + n2;
Xtrain = zeros(nbPoints,nbFeatures);
Xtest = zeros(nbPoints,nbFeatures);
relevance = [0.5*ones(1,nbDummyFeatures),relevance*ones(1,nbRealFeatures)];
associatedValue1 = randi(2,1,nbFeatures) - 1;
associatedValue2 = 1 - associatedValue1;

rtrain = rand(nbPoints,nbFeatures);
rtest = rand(nbPoints,nbFeatures);

for f=1:nbFeatures
    
    % Train
    for p=1:n1
        if (rtrain(p,f) < relevance(f))
            Xtrain(p,f) = associatedValue1(f);
        else
            Xtrain(p,f) = associatedValue2(f);
        end
    end
    for p=((n1+1):(n1+n2))
        if (rtrain(p,f) < relevance(f))
            Xtrain(p,f) = associatedValue2(f);
        else
            Xtrain(p,f) = associatedValue1(f);
        end
    end
    
    % Test
    for p=1:n1
        if (rtest(p,f) < relevance(f))
            Xtest(p,f) = associatedValue1(f);
        else
            Xtest(p,f) = associatedValue2(f);
        end
    end
    for p=((n1+1):(n1+n2))
        if (rtest(p,f) < relevance(f))
            Xtest(p,f) = associatedValue2(f);
        else
            Xtest(p,f) = associatedValue1(f);
        end
    end
end

Ytrain = [ones(n1,1);2*ones(n2,1)];
Ytest = [ones(n1,1);2*ones(n2,1)];

end