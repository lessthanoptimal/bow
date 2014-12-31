function desc = describe_tiny_image( imagePath, N )
  I = imread(imagePath);
  if size(I,3) == 3
    I = rgb2gray(I);
  end
  I    = im2double(I);
  I    = imresize(I,[N N])(:);
  I    = I - mean(I); # make it zero mean
  desc = I ./ norm(I); # make it unit length
end