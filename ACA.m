clear
d = 1;
n = 2^8;
smallest = 2^6;
Lev = log2(n/smallest); % number of levels
k = 15; % target rank
tol = 1e-10;
if d == 1
    f = @(x,y) exp(-abs(x-y).^2);
else
    f = @(x,y) exp(-vecnorm(x'-y').^2);
end
%% Preparing data matrix
rng(5)
x = 4*rand(n,d)-2;

% k-d tree sorting
% number and size of smallest sub-blocks
num = 2^Lev;
for i = 1:d
    for j = 1:num^(i-1)
        siz = n/num^(i-1);
        xsub = sortrows(x((j-1)*siz+1:j*siz,:),i);
        x((j-1)*siz+1:j*siz,:) = xsub;
    end
end

%% Generate A (test purposes)

for i = 1:n
    A(i,:) = f(x(i, :), x(1:n,:));
end
M = A(1:n/2,n/2+1:n);
Mold = M;

%% ACA
Iskel = []; Jskel = [];
% find index of maximum absolute entry
[mvec,ind] = max(abs(M));
[~,j] = max(mvec);
i=ind(j);
Mval = M(i,j); 
% while the absolute entry is greater than tol, repeat
while abs(Mval) > tol
    Iskel = [Iskel, i]; Jskel = [Jskel, j];
    % delete the col and row corresponding the entry, and resume search
    M = M - 1/Mval*M(:,j)*M(i,:);
    % find index of maximum absolute entry
    [mvec,ind] = max(abs(M));
    [val,j] = max(mvec);
    i=ind(j);
    Mval = M(i,j); 
end
norm(Mold- Mold(:,Jskel)*(Mold(Iskel,Jskel)\Mold(Iskel,:)))