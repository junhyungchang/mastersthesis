function xsort = treesort(x, num)
% sort input data x into k-d tree structure
% num is number of leaf level nodes
[n,d] = size(x);
for i = 1:d
    for j = 1:num^(i-1)
        siz = n/num^(i-1);
        xsub = sortrows(x((j-1)*siz+1:j*siz,:),i);
        x((j-1)*siz+1:j*siz,:) = xsub;
    end
end
xsort = x;