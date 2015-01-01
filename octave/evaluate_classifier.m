function evaluate_classifier( funcTrain , funcClassify )
  global verboseFoo
  disp "Loading File Paths"

  [trainSet,labels] = get_image_paths("../brown/data/train");
  [testSet,~] = get_image_paths("../brown/data/test");

  N = 0
  if N > 0
    trainSet = trainSet(randperm(length(trainSet),N));
    testSet = testSet(randperm(length(testSet),N));
  end
  
  disp "Converting Labels"

  trainSet = label_name2index(trainSet,labels);
  testSet = label_name2index(testSet,labels);
  numLabels = length(labels)
  
  disp "Training Tiny"
  classifyData = funcTrain( trainSet , numLabels );
  disp "Test Clasification"
  correct = 0;
  cm = zeros(numLabels,numLabels); # confusion matrix
  countLabel = zeros(numLabels,1);
  for i = 1:length(testSet)
    found = funcClassify( testSet(i).path , classifyData );
    expected = testSet(i).label;
    cm(expected,found) = cm(expected,found) + 1;
    countLabel(expected) = countLabel(expected) + 1;
    if found == expected
      correct = correct + 1;
    end
    if verboseFoo
      disp(sprintf("Correct %5d out of %5d/%5d  guessed %2d actual %2d",correct,i,length(testSet),found,expected))
    end
  end

  # normalize confusion matrix by the number of actual instances
  labelsShort = {};
  for i =1:numLabels
    if countLabel(i) > 0
      cm(i,:) = cm(i,:) ./ countLabel(i);
    end
    labelsShort(end+1) = labels(i){}(1:3);
  end
  
  # plot the confusion matrix
  fig_handle = figure; 
  imagesc(cm, [0 1]); 
  #set(fig_handle, 'Color', [.988, .988, .988])
  axis_handle = get(fig_handle, 'CurrentAxes');
  set(axis_handle, 'XTick', 1:15)
  set(axis_handle, 'XTickLabel', labelsShort)
  set(axis_handle, 'YTick', 1:15)
  set(axis_handle, 'YTickLabel', labels)
  
  
  disp(sprintf("Fraction Correct = %f",correct/length(testSet)))
end