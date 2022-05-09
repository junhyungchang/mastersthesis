clear
% Compare
% 1. KL-expansion approach
% 2. Global low rank approx via randomized eigen decomp.
% 3. HODLR factorization approach
index1 = [2^12];
index2 = [2^18]; % suppress RSVD for index2
d = 1; 
% sq exponential covariance function-----------------------------
% l = .5;
% k = @(x,y) sqexp(x,y,l);

% % exponential kernel----------------------------------------------
% l = 1e+2;
% k = @(x,y) exponential(x,y,l);

% % Matern kernel---------------------------------------------------
l = 2; nu = 3/2;
k = @(x,y) matern(x,y,l,nu); 
%% ---------------------------------------------------------------

t = 2^8; % number of test points
xtest = linspace(-3,3,t)'; % test points
% xtest = 6*rand(t,d)-3;
% xtest = treesort(xtest,2^4);
Rtime = zeros(length(index1),1);
KLtime1 = zeros(length(index1),1);
Htime1 = zeros(length(index1),1);
KLtime2 = zeros(length(index2),1);
Htime2 = zeros(length(index2),1);

%% Smaller n (index1)
count = 0;
for n = index1 % system size (training data)  
    fprintf('n = %d\n', n)
    count = count + 1;
    %% training/test data
    rng(5)
    xtr = 6*rand(n,d)-3;
    xtr = treesort(xtr, 2^5);
    ytr = rand(n,1);
    ytr = ytr/norm(ytr);

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
    %% cross covariance matrix
    Kstar = zeros(t, n);
    for i = 1:t
        Kstar(i,:) = k(xtest(i,:), xtr(:,:));
    end
    % test covariance matrix
    Kstar2 = zeros(t,t);
    for i = 1:t
        Kstar2(i,:) = k(xtest(i,:), xtest(:,:));
    end
% %     K(X,X) (testing)
%     for i = 1:n
%         K(i,:) = k(xtr(i,:),xtr(:,:));
%     end
%     K = K+0.1*eye(n);
    %% Regression via KL-expansion
%     tic
%     m = 35;
%     phi = KLexpansion(m, k);
%     t1 = toc;
%     fprintf('t1:%.4e  ', t1)
%     tic
%     X = zeros(n,m);
%     for j = 1:n
%         X(j,:) = phi(xtr(j,:))';
%     end
%     [U,D,~] = svd(X,'econ');
%     Dinv = zeros(m,m);
%     for i = 1:m
%         Dinv(i,i) = 1/(1+D(i,i)^2);
%     end
%     Kinvy1 = U*(Dinv*(U'*ytr));
%     KLmean = Kstar*Kinvy1;
%     KLmean = real(KLmean);
%     t2 = toc;
%     fprintf('t2:%.4e  ', t2)
%     KLtime1(count) = t1+t2;
%     testv = X*(X'*v);
%     KLerror = norm(Kv - testv(Inde))/norm(Kv);
%     fprintf('KL error: %.4e\n', KLerror)

    %% Regression via Randomized Eigen-decomposition
    % target rank
    r = 35;
    % oversampling parameter
    p = 10; 
    [Ur,Sr, t3, t4]=REig(k, xtr, r, p);
    fprintf('t3:%.4e  t4:%.4e  ', t3, t4)
    tic
    Drinv = zeros(r+p,r+p);
    for i = 1:r+p
        Drinv(i,i) = 1/(Sr(i,i)+0.01);
    end
    Kinvy = Ur*(Drinv*(Ur'*ytr));
    KinvK = Ur*(Drinv*(Ur'*Kstar'));
    Rmean = Kstar*Kinvy;
    t7 = toc;
    fprintf('t7:%.4e  ', t7)
    CovMat = Kstar2-Kstar*KinvK;
    Mean = Rmean;
    VarVec = diag(CovMat);
    Ubd = Mean + 1.96*sqrt(VarVec);
    Lbd = Mean - 1.96*sqrt(VarVec);
    Ubd = real(Ubd);
    Lbd = real(Lbd);
    testv1 = Ur*(Sr*(Ur'*v));
    Rerror = norm(Kv-testv1(Inde))/norm(Kv);
    fprintf('Rerr: %.4e\n', Rerror)
    Rtime(count) = t3 + t4 + t7;

    %% Regression via HODLR factorization
%     siz = 2^7;
%     ell = 35;
%     [y, t5, t6] = hodlr(k, xtr, ytr, siz, ell);
%     fprintf('t5:%.4e  t6:%.4e  ', t5, t6)
%     Hmean = Kstar*y;   
%     Htime1(count) = t5+t6;
%     Mean = Hmean;
%     [KinvK,~,~] = hodlr(k,xtr,Kstar',siz,ell);
%     CovMat = Kstar2 - Kstar*KinvK;
%     VarVec = diag(CovMat);
%     Ubd = Mean + 1.96*sqrt(VarVec);
%     Lbd = Mean - 1.96*sqrt(VarVec);
%     Ubd = real(Ubd);
%     Lbd = real(Lbd);
    %% Print error
%     fprintf('HError: %.4e\n', norm(Rmean-Hmean))
end

% %% Plot comparison of posterior mean
% figure(1)
% plot(xtest, KLmean, 'k', 'LineWidth', 1.5);
% hold on
% plot(xtest, Rmean, 'r.', 'MarkerSize', 9)
% hold on
% plot(xtest, Hmean, 'bo', 'MarkerSize', 7)
% legend('KL-expansion', 'Rand. eig.', 'HODLR')
% title(sprintf('System size n = %d\n Plot of posterior mean on [-3,3]', n))
% hold off
%% Plot for 1-d mean and variance
plot(xtest, Mean, 'b', 'LineWidth', 2)
a=sprintf('Length-scale: %.2f,  nu = %.2f,  N = %d',l,nu, n);
title(a, 'FontSize', 16)
hold on
plot(xtest, Ubd, 'w', 'Linewidth', 0.1)
plot(xtest, Lbd, 'w', 'Linewidth', 0.1)
x2 = [xtest; flipud(xtest)];
inBetween = [Lbd; flipud(Ubd)];
patch(x2, inBetween, 'k');
alpha(0.1)
%% Plot for 2-d mean surface plot
% tri = delaunay(xtest(:,1),xtest(:,2));
% h = trisurf(tri, xtest(:,1),xtest(:,2), Mean);
% axis vis3d
% axis off
% l = light('Position',[-5 -5 5]);
% set(gca,'CameraPosition',[25 -35 .2])
% lighting phong
% shading interp
% colorbar EastOutside



%% Higher n (index2)
% count = 0;
% for n = index2 % system size (training data)  
%     fprintf('n = %d\n', n)
%     count = count + 1;
%     %% training/test data
%     rng(7)
%     xtr = 6*rand([n,1])-3;
%     xtr = treesort(xtr,2^5);
%     ytr = 2*rand([n,1]);
% 
% %     % cross covariance matrix
%     Kstar = zeros(t, n);
%     for i = 1:t
%         Kstar(i,:) = k(xtest(i,:), xtr(:,:));
%     end
% 

%     %% Regression via KL-expansion
%     tic
%     m = 25;
%     phi = KLexpansion(m, k);
%     t1 = toc;
%     fprintf('t1:%.4e  ', t1)
%     tic
%     X = zeros(n,m);
%     for j = 1:n
%         X(j,:) = phi(xtr(j,:))';
%     end
%     [U,D,~] = svd(X,'econ');
%     Dinv = zeros(m,m);
%     for i = 1:m
%         Dinv(i,i) = 1/(1+D(i,i)^2);
%     end
%     KLmean = Kstar*(U*(Dinv*(U'*ytr)));
%     KLmean = real(KLmean);
%     t2 = toc;
%     fprintf('t2:%.4e  ', t2)
%     KLtime2(count) = t1+t2;

%     %% Regression via HODLR factorization
%     siz = 2^6;
%     ell = 25;
%     [y, t5, t6] = hodlr(k, xtr, ytr, siz, ell);
%     fprintf('t5:%.4e  t6:%.4e\n', t5, t6)
%     Hmean = Kstar*y;   
%         Htime2(count) = t5+t6;
% 
% %     %% Print times
% %     fprintf('%.4f %.4f %.4f %.4f %.4f %.4f\n', t1, t2, t5, t6)
% end


% %% Plot comparison of posterior mean
% figure(2)
% plot(xtest, KLmean, 'k', 'LineWidth', 1.5);
% hold on
% plot(xtest, Hmean, 'bo', 'MarkerSize', 7)
% legend('KL-expansion', 'HODLR')
% title(sprintf('System size n = %d\n Plot of posterior mean on [-3,3]', n))
% hold off

% %% Plot times
% 
% figure(3)
% KLtime = [KLtime1', KLtime2'];
% Htime = [Htime1', Htime2'];
% index22 = [index1, index2];
% loglog(index1, Rtime, '-sr', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
% hold on
% loglog(index22, KLtime, '-sb', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
% hold on
% loglog(index22, Htime, '-sk', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor',[0 0 0])
% title('Run-time comparison')
% xlim([2^6, 2^21])
% xticks([10^2 10^4 10^6])
% xticklabels({'10^2', '10^4', '10^6'}) 
% xlabel('System size')
% ylabel('elapsed time (seconds)')
% 
% 
