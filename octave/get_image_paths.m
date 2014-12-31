function [image_paths,labelNames] = get_image_paths( rootDir )
  image_paths=struct([]);
  labels = find_label_dirs( rootDir );
  labelNames = {};
  for i = 1:length(labels)
    a = labels(i);
    labelNames(end+1) = a.label;
    listing = dir(a.path);
    for j = 1:length(listing)
      f = listing(j);
      if !f.isdir && strendswith(f.name,".jpg")
	      c = struct;
	      c.path = strcat(a.path,"/",f.name);
	      c.label = a.label;
	      image_paths(end+1) = c;
      end
    end
  end
  
  
      
function labels = find_label_dirs( rootDir )
  labels = struct([]);
  listing = dir(rootDir);
  for i = 1:length(listing)
    d = listing(i);
    if d.isdir && not(d.name(1) == '.')
      l=struct;
      l.path = strcat(rootDir,"/",d.name);
      l.label = d.name;
      labels(end+1) = l;
    end
  end
  
