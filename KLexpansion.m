function phi = KLexpansion(m, k)
% Code for KL-expansion on interval [-b,b]
% m is length of expansion
% k is covariance function
% returns a length-m function handle containing the KL basis functions
b=3;
[xleg,w] = legpts(m);
xleg = xleg*b;
A = zeros(m,m);
for i = 1:m
    for j = 1:m
        A(i,j) =sqrt(w(i)*w(j))*k(xleg(i), xleg(j));
    end
end
[U, D] = eig(A);
for i = 1:m
    U(i,:) = U(i,:)/sqrt(w(i));
end

M = zeros(m,m);

for i = 1:m
%     M(i,:) = legendreP(i-1, xleg/b);
%     M(i,:) = M(i,:).*w;
    M(:,i) = legendreP(i-1, xleg/b);
end
% % coefficients of Legendre poly approximation
% C = M*U;
C = M\U;

% find Legendre poly approx function
syms x
P = legendreP(0:m-1, x);
P = matlabFunction(P);
P = @(x) P(x/b);
ux = @(x) dot(C(:,1), P(x));
for j = 2:m
    ux = @(x) [ux(x); dot(C(:,j), P(x))];
end
L = diag(D);
phi = @(x) sqrt(L).*ux(x);
% o = 15; r = 15; q = 9.559790239932053e-01;
% fprintf('%.6e\n', k(q,q)- sum(phi(q).*phi(q)))


