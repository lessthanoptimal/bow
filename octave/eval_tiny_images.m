more off
disp "Loading File Paths"

[trainSet,labels] = get_image_paths("../brown/data/train");
[testSet,a] = get_image_paths("../brown/data/test");

#trainSet = trainSet(1:30);

disp "Converting Labels"

trainSet = label_name2index(trainSet,labels);
testSet = label_name2index(testSet,labels);

global verboseFoo = 1
global imageWidth = 16
global numNeighbors = 40
global numLabels = length(labels)

function tinyData = trainTinyClassifier( dataSet )
  global imageWidth
  global verboseFoo
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

disp "Training Tiny"
tinyData = trainTinyClassifier( trainSet );
disp "Test Clasification"
correct = 0;
for i = 1:length(testSet)
  found = tinyClassifier( testSet(i).path , tinyData );
  expected = testSet(i).label;
  if found == expected
    correct = correct + 1;
  end
  if verboseFoo
    disp(sprintf("Correct %5d out of %5d/%5d",correct,i,length(testSet)))
  end
end

disp(sprintf("Fraction Correct = %f",correct/length(testSet)))
disp(sprintf("K-NN = %d width = %d",numNeighbors,imageWidth))

printf "DONE!\n"

# Fraction Correct = 0.233166 K = 1
# Fraction Correct = 0.223116 K = 10
# Fraction Correct = 0.196985 K = 40
  