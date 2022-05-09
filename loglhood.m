function [f, g] = loglhood(z)
% compute log likelihood and gradient with RBF kernel
% sq exponential covariance function-----------------------------
d = 1; 
n = 2^12;
% fprintf('n:%d \n',n)
rng(7)
xtr = 6*rand(n,d)-3;
xtr = treesort(xtr, 2^5);
ytr = 2*rand(n,1);

k = @(x,y) sqexp(x,y,z);
[n,d] = size(xtr);
for i = 1:n
    C(i,:) = k(xtr(i,:),xtr(:,:));
end
C = C + eye(n);
Cinvy = C\ytr;
f = -1/2*ytr'*Cinvy -1/2*log(det(C))-n/2*log(2*pi);
f= -f;
% gradient --------------------------------------
kdiff = @(x,y) sqexpdiff(x,y,z);
for i = 1:n
    Ci(i,:) = kdiff(xtr(i,:),xtr(:,:));
end
CCinvy = Ci*Cinvy;
CiCCi = Cinvy'*CCinvy;
trterm = trace(C\Ci);
g = -1/2*CiCCi-1/2*trterm;
g = -g;
end
