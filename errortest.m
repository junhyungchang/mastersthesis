clear
% Compare
% 1. KL-expansion approach
% 2. Global low rank approx via randomized eigen decomp.
% 3. HODLR factorization approach

%% Kernels
% sq exponential covariance function-----------------------------
l = 1;
k = @(x,y) sqexp(x,y,l);

% % exponential kernel----------------------------------------------
% l = 1e+2;
% k = @(x,y) exponential(x,y,l);

% %Matern kernel---------------------------------------------------
% l = 1; nu = 5/2;
% k = @(x,y) matern(x,y,l,nu); 
%% Setup ----------------------------------------------------------
d = 1; 
n = 2^8;
fprintf('n:%d \n',n)
rng(7)
xtr = 6*rand(n,d)-3;
xtr = treesort(xtr, 2^5);
ytr = 2*rand(n,1);
% target rank
r = 20;
% oversampling parameter
p = 5; 
% Length of KL expansion
m = 35;
%% Generate test vector
v = rand(n,1); v = v/norm(v);
% compute K*v
if n < 2^15
    Inde = 1:n;
    for i = 1:n
%         K(i,:) = k(xtr(i,:), xtr(:,:));
        if d == 1
            Krow = k(xtr(i,:),xtr(:,:))';
            Kv(i) = Krow*v;
        else 
            Krow = k(xtr(i,:),xtr(:,:));
            Kv(i) = Krow*v;
        end
    end
else % when n becomes too large to compute Kv quickly
    Inde = randi([1,n],1,1000);
    Inde = sort(Inde);
    for i = 1:length(Inde)
        if d == 1
            Krow = k(xtr(Inde(i),:),xtr(:,:))';
            Kv(i) = Krow*v;
        else 
            Krow = k(xtr(Inde(i),:),xtr(:,:));
            Kv(i) = Krow*v;
        end
    end
end
Kv = Kv';
% %% Regression via KL-expansion

% fprintf('length: %d\n', m)
% phi = KLexpansion(m, k);
% X = zeros(n,m);
% for j = 1:n
%     X(j,:) = phi(xtr(j,:))';
% end
% testv = X*(X'*v);
% KLerror = norm(Kv - testv(Inde))/norm(Kv);
% fprintf('KL error: %.4e\n', KLerror)

% %% direct REig
% fprintf('r+p: %d\n', r+p)
% [Ur,Sr, t3, t4]=REig(k, xtr, r, p);
% testv1 = Ur*(Sr*(Ur'*v));
% Rerror = norm(Kv-testv1(Inde))/norm(Kv);
% fprintf('Rerr: %.4e\n', Rerror)

%% Regression via HODLR factorization
siz = 2^6;
ell = 25;
[y, t5, t6] = hodlr(k, xtr, v, siz, ell);
Herror = norm(Kv-y(Inde))/norm(Kv);
fprintf('Herror: %.4e\n', Herror)