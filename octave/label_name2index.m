function dataSet = label_name2index( dataSet , labels )
  for i = 1:length(dataSet)
    dataSet(i).label = name2index(dataSet(i).label,labels);
  end
end

function index = name2index(name,labels)
  for i = 1:length(labels)
    if( strcmpi(name,labels(i){} ))
      index = i;
      return
    end
  end
  disp "Oh Shit!  Can't convert label to index"
  exit(0)
end
     