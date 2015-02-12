# Works with Octave 3.8.1 and VLFeat 0.9.19
# Classifier which uses SIFT histogram and linear SVM with 1-vs-All classifier for type

more off
addpath "/home/pja/projects/thirdparty/vlfeat-0.9.19/toolbox"
vl_setup

global verboseFoo = 1
global kmeansClusters = 400
global kmeansSeeds = 1
global maxFeaturesPerImage = 150
global smoothInput = 0 # magnification.  try 3.  disable <= 0

global lambda = 0.0001 ; % Regularization parameter
global maxIter = 100000 ; % Maximum number of iterations
global numSvnTrainSeeds = 200;

function siftData = trainSiftLinearSVM( trainSet , numLabels )
  global verboseFoo
  global kmeansClusters
  global kmeansSeeds
  global smoothInput
  global maxFeaturesPerImage
  global lambda
  global maxIter
  global numSvnTrainSeeds

  nameClusters = sprintf("cluster_K%0d.data",kmeansClusters);
  nameHistograms = sprintf("histograms_K%0d.data",kmeansClusters);

  # create the clusters

  if exist(nameClusters)
    disp(sprintf("Loading %s",nameClusters));
    clusters = load(nameClusters).clusters;
  else
    disp("Computing clusters")
    clusters = clusterSiftDesc(trainSet,kmeansClusters,maxFeaturesPerImage,verboseFoo,kmeansSeeds,smoothInput);
    save(nameClusters,"clusters");
  end

  # compute histograms from clusters

  if exist(nameHistograms)
    disp(sprintf("Loading %s",nameHistograms));
    imageInfo = load(nameHistograms).imageInfo;
  else
    disp("Computing histograms from clusters")
    imageInfo=struct([]);
    for i = 1:length(trainSet)
      s = struct;
      s.hist = describeSiftHistogram(trainSet(i).path,clusters,smoothInput);
      s.label = trainSet(i).label;
      imageInfo(end+1) = s;
    end
    save(nameHistograms,"imageInfo");
  end

  # train SVN classifiers
  machines = struct([]);
  for i = 1:numLabels
    disp(sprintf("Training SVM %d",i))
    X = []; Y = [];
    for j = 1:length(imageInfo)
      s = imageInfo(j);
      X = [X,s.hist];
      if i == s.label
        Y(end+1) = 1;
      else
        Y(end+1) = -1;
      end
    end
    bestFMeas = 0;
    for j=1:numSvnTrainSeeds
      [w b info] = vl_svmtrain(X, Y, lambda, 'MaxNumIterations', maxIter, 'Epsilon',1e-4);
      [~,~,~, scores] = vl_svmtrain(X, Y, lambda, 'model', w, 'bias', b, 'solver', 'none');
      TP = sum(scores(Y==1)>0);
      FP = sum(scores(Y==0)>0);
      FN = sum(scores(Y==1)<0);
      precision = TP/(TP+FP);
      recall = TP/(TP+FN);
      Fmeas = 2*(precision*recall)/(precision+recall);
      if recall > bestFMeas
        bestFMeas = recall;
        bestW = w;
        bestB = b;
        bestInfo = info;
      end
      #disp(sprintf("SVM Training[%2d] = %f",j,Fmeas))
    end

    svmParam = struct;
    svmParam.w = bestW;
    svmParam.b = bestB;
    svmParam.info = bestInfo;
    machines(end+1) = svmParam;
    [~,~,~, scores] = vl_svmtrain(X, Y, lambda, 'model', bestW, 'bias', bestB, 'solver', 'none') ;
    disp(sprintf("   total true correct = %d out of %d",sum(scores(Y==1)>0),sum(Y==1)))
  end

  #keyboard

  siftData.clusters = clusters;
  siftData.numLabels = numLabels;
  siftData.images = imageInfo;
  siftData.machines = machines;
end

function results = classifierSiftLinearSVM( imagePath , siftData )
  global smoothInput
  global kmeansClusters

  D = describeSiftHistogram(imagePath,siftData.clusters,smoothInput);

  # run it through each SVM and pick the best one with the highest likelihood
  bestScore = 0;
  bestLabel = -1;
  for i = 1:siftData.numLabels
    svmParam = siftData.machines(i);
    score = svmParam.w'*D + svmParam.b; # bias term could be ignored
    if score > bestScore || bestLabel == -1
      bestScore = score;
      bestLabel = i;
    end
    # disp(sprintf("score %d = %f",i,score))
  end

  results = bestLabel;
end

evaluate_classifier(@trainSiftLinearSVM,@classifierSiftLinearSVM)

disp(sprintf("lambda = %f clusters = %3d  maxPerImage = %3d smooth = %d", ...
  lambda,kmeansClusters,maxFeaturesPerImage,smoothInput))

# 0.673367  lambda = 0.000100 clusters = 400  maxPerImage = 150
# 0.676047  lambda = 0.000100 clusters = 400  maxPerImage = 150
# 0.624791  lambda = 0.001000 clusters = 400  maxPerImage = 150