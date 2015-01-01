more off

global verboseFoo = 1
global imageWidth = 16
global numNeighbors = 5
global numLabels

function tinyData = trainTinyClassifier( dataSet , numLabels_ )
  global imageWidth
  global verboseFoo
  global numLabels
  
  numLabels = numLabels_
  tinyData = struct([]);
  for i = 1:length(dataSet)
    d = dataSet(i);
    d.desc = describe_tiny_image(d.path,imageWidth);
    tinyData(end+1) = d;
    if verboseFoo ~= 0
      disp(sprintf("train with %s",d.path))
    end
  end
end

function results = tinyClassifier( imagePath , tinyData )
  global imageWidth
  global numLabels
  global numNeighbors

  target = describe_tiny_image(imagePath,imageWidth);

  scores = zeros(length(tinyData),1);
  for i = 1:length(tinyData)
    d = tinyData(i).desc;
    scores(i) = norm(d-target);
  end
  [~,I] = sort(scores);
  
  # see what the most common label is in the K best fits
  hist=zeros(numLabels,1);
  for i = 1:numNeighbors
    label = tinyData(I(i)).label;
    hist(label) = hist(label) + 1;
  end
  
  [~,results] = max(hist);
end

evaluate_classifier(@trainTinyClassifier,@tinyClassifier)

disp(sprintf("K-NN = %d width = %d",numNeighbors,imageWidth))

printf "DONE!\n"

# Fraction Correct = 0.233166 K = 1
# Fraction Correct = 0.228141 K = 5
# Fraction Correct = 0.223116 K = 10
# Fraction Correct = 0.196985 K = 40
  