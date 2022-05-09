function [U,V] = RLR(f, x1, x2, r, p)
% Randomized Low-Rank factors A \approx U*V'
% Double-pass algorithm for general SQUARE matrices
% f is function handle for entry evaluation
% x is the vector of training points to evaluate at
% r is the target rank
% p is the oversapling parameter
% r = 10, p = 10 yields good results

[m,d] = size(x1);
G = randn(m, r+p);

Y = zeros(m,r+p);
for i = 1:m
    if d == 1
        Arow = f(x1(i,:),x2(:,:))';
    else
        Arow = f(x1(i,:),x2(:,:));
    end
    Y(i, :) = Arow*G;
end
% for i = 1:m
%     A = f(x1(i), x2(1:m))';
%     Y(i, 1:r+p) = A*G;
% end
[Q,~] = qr(Y,0);
C = zeros(r+p,m);
for i = 1:m
    if d == 1
        Acol = f(x1(:,:),x2(i,:));
    else 
        Acol = f(x1(:,:),x2(i,:))';
    end
    C(:,i) = Q'*Acol;
end
% for i = 1:m
%     A = f(x1(1:m), x2(i));
%     C(1:r+p, i) = Q'*A;
% end
[Uhat, D, Vhat] = svd(C, 'econ');
U = Q;
V = Vhat*D*Uhat';
% size(U)
% rank(U)
% size(V)
% rank(V)
