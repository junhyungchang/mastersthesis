function [U, V] = ACALR(f, x1, x2, r)
%% Low rank approx via ACA
% returns U,V such that A approx U*V'
m = length(x1);

for i = 1:m
    M(i,:) = f(x1(i,:), x2(1:m,:));
end
Mold = M;
Iskel = []; Jskel = [];
% find index of maximum absolute entry
[mvec,ind] = max(abs(M));
[~,j] = max(mvec);
i=ind(j);
Mval = M(i,j); 
% while the absolute entry is greater than tol, repeat
% while abs(Mval) > tol
while length(Jskel) < r
    Iskel = [Iskel, i]; Jskel = [Jskel, j];
    % delete the col and row corresponding the entry, and resume search
    M = M - 1/Mval*M(:,j)*M(i,:);
    % find index of maximum absolute entry
    [mvec,ind] = max(abs(M));
    [~,j] = max(mvec);
    i=ind(j);
    Mval = M(i,j); 
end
U = Mold(:,Jskel);
V = (Mold(Iskel,Jskel)\Mold(Iskel,:))';
% norm(Mold- Mold(:,Jskel)*(Mold(Iskel,Jskel)\Mold(Iskel,:)))