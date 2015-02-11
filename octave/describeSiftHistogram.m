# Computes the histogram for an image by computing SIFT features and matching to clusters
#
# imagePath Path to the image
# clusters List of clusters
function desc = describeSiftHistogram( imagePath , clusters , smoothInput )

  # Dense computation of SIFT features
  D = describeImageSift(imagePath,-1,smoothInput);

  # Find best matches to each sift feature in the clusters
  I = vl_ikmeanspush(D,clusters);

  #compute the histogram
  numClusters = size(clusters,2);
  desc = vl_ikmeanshist(numClusters,I);

  # L2-normalization of histogram
  desc = desc ./ norm(desc);
end