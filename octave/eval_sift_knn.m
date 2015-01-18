# Works with Octave 3.8.1 and VLFeat 0.9.19
# Classifier which uses SIFT histogram and K-Nearest-Neighbor classifier

more off
addpath "/home/pja/projects/thirdparty/vlfeat-0.9.19/toolbox"
vl_setup

global verboseFoo = 1
global numNeighbors = 5
global kmeansClusters = 200
global kmeansSeeds = 20
global maxfeaturesPerImage = 150
global smoothInput = 0 # magnification.  try 3.  disable <= 0 

function desc = describeImageSift( imagePath , numFeatures )
  global smoothInput
  
  binSize = 4;
  step = 10;

  I = imread(imagePath);
  if size(I,3) == 3
    I = rgb2gray(I);
  end
  I = single(I);
  
  # pre-smooth image  
  if smoothInput > 0
   I = vl_imsmooth(I, sqrt((binSize/smoothInput)^2 - .25)) ;
  end
  [~,D] = vl_dsift(I,'size',binSize,'step',step);
  M = size(D,2);

  if numFeatures > 0 && numFeatures < M
    N = min(numFeatures,M);
    indexes = 1:size(D,2);
    desc = D(:,randperm(M,N));
  else
    desc = D;
  end
end

function error = clusterError( descAll , I , cluster )
  error = 0;
  
  for i = 1:size(descAll,2)
    d = descAll(:,i);
    c = cluster(:,I(i));
    error = error + norm(double(d)-double(c));
  end
end  

function clusters = clusterSiftDesc( trainSet )
  global kmeansClusters
  global maxfeaturesPerImage
  global verboseFoo
  global kmeansSeeds
  
  # compute the descriptions for all input images
  descAll = [];
  
  for i=1:length(trainSet)
    if verboseFoo
      disp(sprintf("cluster %4d of %4d",i,length(trainSet)))
    end
    desc = describeImageSift(trainSet(i).path,maxfeaturesPerImage);
    descAll = [descAll , desc];
  end
   
  # run k-means to find clusters
  if verboseFoo
    disp("Computing Clusters")
  end
  [clusters,I] = vl_ikmeans(descAll,kmeansClusters);
  
  # Try to find a better cluster by minimizing the distance
  if kmeansSeeds > 1
    bestClusters = clusters;
    bestError = clusterError(descAll,I,clusters);
    for i=2:kmeansSeeds
      [clusters,I] = vl_ikmeans(descAll,kmeansClusters);
      error = clusterError(descAll,I,clusters);
      if error < bestError
        bestError = error;
        bestClusters = clusters;
      end
      
      if verboseFoo
        disp(sprintf("cluster error [%2d] = %8.1f best %8.1f",i,error,bestError))
      end
    end
  end  
end

function desc = describeSiftHistogram( imagePath , clusters )
  global kmeansClusters

  D = describeImageSift(imagePath,-1);
  I = vl_ikmeanspush(D,clusters);
  desc = vl_ikmeanshist(kmeansClusters,I);
  desc = desc ./ sum(desc);
end  

function siftData = trainSiftKMeans( trainSet , numLabels )
  global verboseFoo
  global kmeansClusters
  
  nameClusters = sprintf("cluster_K%0d.data",kmeansClusters);
  nameHistograms = sprintf("histograms_K%0d.data",kmeansClusters);
    
  if exist(nameClusters)
    disp(sprintf("Loading %s",nameClusters));
    clusters = load(nameClusters).clusters;
  else
    disp("Computing clusters")
    clusters = clusterSiftDesc(trainSet);
    save(nameClusters,"clusters"); 
  end
  
  if exist(nameHistograms)
    disp(sprintf("Loading %s",nameHistograms));
    imageInfo = load(nameHistograms).imageInfo;
  else    
    disp("Computing histograms from clusters")
    imageInfo=struct([]);
    for i = 1:length(trainSet)
      s = struct;
      s.hist = describeSiftHistogram(trainSet(i).path,clusters);
      s.label = trainSet(i).label;
      imageInfo(end+1) = s;
    end
    save(nameHistograms,"imageInfo"); 
  end
  siftData.clusters = clusters;
  siftData.numLabels = numLabels;
  siftData.images = imageInfo;
end

function results = classifierSiftKMeans( imagePath , siftData )
  global numNeighbors

  D = describeSiftHistogram(imagePath,siftData.clusters);
  
  # See which images it's most similar to
  errors = zeros(length(siftData.images),1);
  
  for i = 1:length(siftData.images)
    image = siftData.images(i);
    errors(i) = norm(image.hist-D);
  end
  
  # Vote on the label
  [~,I] = sort(errors);
  
  hist = zeros(siftData.numLabels,1);
  for i = 1:numNeighbors
    label = siftData.images(I(i)).label;
    hist(label) = hist(label) + 1;
  end

  [~,results] = max(hist);
end

# TODO explore better clusters.  Recomute seeds for better solution
# TODO explore other clustering algorithms
# TODO change SIFT sampling

evaluate_classifier(@trainSiftKMeans,@classifierSiftKMeans)

disp(sprintf("K-NN = %2d clusters = %3d  maxPerImage = %3d smooth = %d", ...
  numNeighbors,kmeansClusters,maxfeaturesPerImage,smoothInput))

# 0.497487  K-NN =  5 clusters = 50   maxPerImage = 150
# 0.492462  K-NN = 15 clusters = 50   maxPerImage = 150
# 0.515243  K-NN =  5 clusters = 200  maxPerImage = 150
# 0.500168  K-NN = 15 clusters = 200  maxPerImage = 150
# 0.506198  K-NN = 15 clusters = 200  maxPerImage = 150 <-- clusters optimized
# 0.486767  K-NN =  5 clusters = 400  maxPerImage = 150

# 0.454271  K-NN = 5 clusters = 50    maxPerImage = 150 smooth = 3