# Use VL-Feat's k-means clustering algorithm to compute cluster centers which will act as visual
# words in the training set.  VLFeat also computes the SIFT features
#
# trainSet list of image files paths
# kmeansClusters number of clusters
# kmeansSeeds Number of times it will run the clustering algorithms.
# smoothInput should it smooth the input image or not

function clusters = clusterSiftDesc( trainSet , kmeansClusters , \
				    maxfeaturesPerImage, verboseFoo, \
				    kmeansSeeds ,smoothInput)

  # compute the descriptions for all input images
  descAll = [];
  
  for i=1:length(trainSet)
    if verboseFoo
      disp(sprintf("cluster: describe file %4d of %4d",i,length(trainSet)))
    end
    desc = describeImageSift(trainSet(i).path,maxfeaturesPerImage,smoothInput);
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