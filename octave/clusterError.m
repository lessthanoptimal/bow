function error = clusterError( descAll , I , cluster )
  error = 0;

  for i = 1:size(descAll,2)
    d = descAll(:,i);
    c = cluster(:,I(i));
    error = error + norm(double(d)-double(c));
  end
end