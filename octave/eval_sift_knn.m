# Works with Octave 3.8.1 and VLFeat 0.9.19
# Classifier which uses SIFT histogram and K-Nearest-Neighbor classifier

more off
addpath "/home/pja/projects/thirdparty/vlfeat-0.9.19/toolbox"
vl_setup

global verboseFoo = 1
global numNeighbors = 5
global kmeansClusters = 200
global kmeansSeeds = 1
global maxFeaturesPerImage = 150
global smoothInput = 0 # magnification.  try 3.  disable <= 0

function siftData = trainSiftKMeans( trainSet , numLabels )
  global verboseFoo
  global kmeansClusters
  global kmeansSeeds
  global smoothInput
  global maxFeaturesPerImage
  
  nameClusters = sprintf("cluster_K%0d.data",kmeansClusters);
  nameHistograms = sprintf("histograms_K%0d.data",kmeansClusters);
    
  if exist(nameClusters)
    disp(sprintf("Loading %s",nameClusters));
    clusters = load(nameClusters).clusters;
  else
    disp("Computing clusters")
    clusters = clusterSiftDesc(trainSet,kmeansClusters,maxFeaturesPerImage,verboseFoo,kmeansSeeds,smoothInput);
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
      s.hist = describeSiftHistogram(trainSet(i).path,clusters,smoothInput);
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
  global smoothInput

  D = describeSiftHistogram(imagePath,siftData.clusters,smoothInput);

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

# TODO explore better clusters.  Recompute seeds for better solution
# TODO explore other clustering algorithms
# TODO change SIFT sampling

evaluate_classifier(@trainSiftKMeans,@classifierSiftKMeans)

disp(sprintf("K-NN = %2d clusters = %3d  maxPerImage = %3d smooth = %d", ...
  numNeighbors,kmeansClusters,maxFeaturesPerImage,smoothInput))

# 0.497487  K-NN =  5 clusters = 50   maxPerImage = 150
# 0.492462  K-NN = 15 clusters = 50   maxPerImage = 150
# 0.515243  K-NN =  5 clusters = 200  maxPerImage = 150
# 0.500168  K-NN = 15 clusters = 200  maxPerImage = 150
# 0.506198  K-NN = 15 clusters = 200  maxPerImage = 150 <-- clusters optimized
# 0.486767  K-NN =  5 clusters = 400  maxPerImage = 150

# 0.454271  K-NN = 5 clusters = 50    maxPerImage = 150 smooth = 3