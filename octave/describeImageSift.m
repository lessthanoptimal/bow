# Computes a dense set of SIFT features from the specified image
#
# imagePath where the image is
# numFeatures The maximum number of features it will return.  If <= 0 then all features are returned.
#             if there are more features than numFeatures than numFeatures will be randomly selected
# smoothInput Should it apply Gaussian smooth to the input image or not

function desc = describeImageSift( imagePath , numFeatures , smoothInput )
  if ~exist("smoothInput")
    smoothInput = 0;
  end

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

