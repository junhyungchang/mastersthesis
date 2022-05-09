clear
% Change to inverse, add levels
d = 1;
% sq exponential covariance function-----------------------------
l = 1;
f = @(x,y) sqexp(x,y,l);

% % exponential kernel----------------------------------------------
% l = 1e+2;
% f = @(x,y) exponential(x,y,l);

% % Matern kernel---------------------------------------------------
% l = 1; nu = 5/2;
% f = @(x,y) matern(x,y,l,nu); 
% ---------------------------------------------------------------
n = 2^8;
Lev = 2; % number of levels
k = 12; % target rank
%% Preparing data matrix
rng(5)
x = 6*rand(n,d)-3;

% k-d tree sorting
% number and size of smallest sub-blocks
num = 2^Lev;
% call treesort function
x = treesort(x, num);

% Skeleton index in the last row of the data matrix
x = [x, ones(n,1)];

% % Plot the DOFs
% figure(1)
% for i = 1:n
%     if x(i,d+1) == 1
%         plot(x(i,1), x(i,2), 'b.', 'MarkerSize', 7)
%         hold on
%     end
% end
% hold off
%% Recursive Skeletonization Factorization

PP = [];
Qi = [];
Si = [];
DD = sparse(n,n);
Order = [1:n];

% Loop over levels
tic
% Level 0
for c = 1
    m = n/2^Lev;
    Pis = [];
    Qinvs = [];
    Sinvs = [];
    % Loop over subdomains
    for z = 1:n/m
        % Location of block
        p1 = n-z*m;
        
        % Evaluate necessary sub-blocks
        % Aps and Aqr blocks
        if z == 1
            ind1 = 1:p1;
            ind2 = p1+1:n;
        elseif z == n/m
            ind1 = m+1:n;
            ind2 = 1:m;
        else
            ind1 = [1:p1, p1+m+1:n];
            ind2 = p1+1:p1+m;
        end
        for i = 1:length(ind1)
            if x(Order(ind1(i)), d+1) == 0
                Apsqr(i,:) = zeros(1,m);
            else
                Apsqr(i,:) = f(x(Order(ind1(i)),1:d), x(ind2,1:d));
            end
        end
        
        % Skeletonization
        % Take advantage of the sparsity
        I = find(all(Apsqr~=0, 2));
        Amini = Apsqr(I,:);
        [S, T] = IDQR(Amini, k);

        % permutation matrix for skeleton order
        SS = sparse(m,m);
        Pmt = speye(n);
        for i = 1:m
            SS(S(i), i) = 1;
        end
        Pmt(p1+1:p1+m, p1+1:p1+m) = SS;
        Apsqr = Apsqr*SS;
        Aoff = Apsqr(:, 1:k);
        % Keep track of the ordering of DOFs
        Order = Order*Pmt;
        
        % evaluate block of size m x m 
        for i = 1:m
            B(i,:) = f(x(p1+i,1:d), x(ind2,1:d));
        end
        B = SS'*(B*SS);
        B = B + eye(m);
        
        % evaluate necessary blocks from B
        Ass = B(1:k, 1:k);
        Asr = B(1:k, k+1:m);
        Ars = Asr';
        Arr = B(k+1:m, k+1:m);

        Brr = Arr - T'*Asr - Ars*T + T'*Ass*T;
        Bsr = Asr - Ass*T;
        Brs = Bsr';
        Bss = Ass - Bsr*(Brr\Brs);

        % LDL decomposition of Brr
        [L,D] = ldl(Brr);

        % Compute block LU factors
        
        Pis = [Pis, Pmt'];
%         tv1 = Pmt'*tv;

        Qinv = speye(n);
        Qinv(p1+1:p1+k,p1+k+1:p1+m) = T;
%         tv1 = Qinv*tv1;
        Qinvs = [Qinvs, Qinv]; 
    
        S1inv = speye(n);
        S1inv(p1+k+1:p1+m,p1+k+1:p1+m) = L';
        S2inv = speye(n);
        S2inv(p1+k+1:p1+m,p1+1:p1+k) = D\(L\Brs);
        Sinv = S2inv*S1inv;
        Sinvs = [Sinvs, Sinv];
%         tv1 = S2inv*(S1inv*tv1);
        
        % Introduce sparsity in DD to block row and column
        DD = Pmt'*DD*Pmt;
        DD(p1+k+1:p1+m,ind1) = sparse(m-k,n-m);
        DD(ind1,p1+k+1:p1+m) = sparse(n-m,m-k);
        
        
        DD(ind1, p1+1:p1+k) = Aoff;
        DD(p1+1:p1+k, ind1) = Aoff';    
        
        DD(p1+1:p1+k,p1+1:p1+k) = Bss;
        % This will change for higher levels
        DD(p1+k+1:p1+m,p1+k+1:p1+m) = D;
        
        % retrieve indices of the inputs in the residual set
%         pind = [1:n]*Pmt;
        rind = Order(p1+k+1:p1+m);  
        % switch skeleton index to 0 if in residual set
        x(rind, end) = zeros(length(rind), 1);

    end
    PP = [PP; Pis];
    Qi = [Qi; Qinvs];
    Si = [Si; Sinvs];
end
% % Spy DD
% figure(1)
% spy(DD)
% xlabel('')
% % Plot the DOFs
% figure(2)
% for i = 1:n
%     if x(i,d+1) == 1
%         plot(x(i,1), x(i,2), 'b.', 'MarkerSize', 10)
%         hold on
%     end
% end
% hold off
% Higher levels -------------------------------------------
for c = 2:Lev
    m = n/2^Lev*2^(c-1);
    Pis = [];
    Qinvs = [];
    Sinvs = [];

    % Loop over subdomains
    for z = 1:n/m
        % Location of block
        p1 = n-z*m;
        
        % For easier indexing
        % ind2 is the index for m x m block of interest
        % ind1 is the index for the rest
        if z == 1
            ind1 = 1:p1;
            ind2 = p1+1:n;
        elseif z == n/m
            ind1 = m+1:n;
            ind2 = 1:m;
        else
            ind1 = [1:p1, p1+m+1:n];
            ind2 = p1+1:p1+m;
        end
        % Reorder m x m blocks
        Pmt1 = speye(n);
        SS1 = speye(m);
        tempcol = SS1(:,k+1:m/2);
        SS1(:, k+1:2*k) = SS1(:,m/2+1:m/2+k);
        SS1(:,2*k+1:m/2+k) = tempcol;
        Pmt1(ind2,ind2)= SS1;
        DD = Pmt1'*DD*Pmt1;
        Order = Order*Pmt1;
        % New Apsqr blocks
        Apsqr = full(DD(ind1, p1+1:p1+2*k));
        
        % Skeletonization
        I = find(all(Apsqr~=0, 2));
        Amini = Apsqr(I,:);
        [S, T] = IDQR(Amini, k);


        % permutation matrix for skeleton order
        SS = sparse(2*k,2*k);
        Pmt = speye(n);
        for i = 1:2*k
            SS(S(i), i) = 1;
        end
        Pmt(p1+1:p1+2*k, p1+1:p1+2*k) = SS;
        Apsqr = Apsqr*SS;
        Aoff = Apsqr(:, 1:k);
        % Keep track of the ordering of DOFs
        Order = Order*Pmt;
        
        B = DD(p1+1:p1+2*k,p1+1:p1+2*k);
        B = SS'*(B*SS);
%         B = B + eye(m);
        
        % evaluate necessary blocks from B
        Ass = B(1:k, 1:k);
        Asr = B(1:k, k+1:2*k);
        Ars = Asr';
        Arr = B(k+1:2*k, k+1:2*k);

        Brr = Arr - T'*Asr - Ars*T + T'*Ass*T;
        Bsr = Asr - Ass*T;
        Brs = Bsr';
        Bss = Ass - Bsr*(Brr\Brs);
        Brr = full(Brr);
        % LDL decomposition of Brr
        [L,D] = ldl(Brr);

        % Compute block LU factors
        Pmts = Pmt'*Pmt1';
        Pis = [Pis, Pmts];
%         tv1 = Pmt'*tv;

        Qinv = speye(n);
        Qinv(p1+1:p1+k,p1+k+1:p1+2*k) = T;
%         tv1 = Qinv*tv1;
        Qinvs = [Qinvs, Qinv]; 
    
        S1inv = speye(n);
        S1inv(p1+k+1:p1+2*k,p1+k+1:p1+2*k) = L';
        S2inv = speye(n);
        S2inv(p1+k+1:p1+2*k,p1+1:p1+k) = D\(L\Brs);
        Sinv = S2inv*S1inv;
        Sinvs = [Sinvs, Sinv];
%         tv1 = S2inv*(S1inv*tv1);
        
        % Introduce sparsity in DD to block row and column
        DD = Pmt'*DD*Pmt;
        DD(p1+k+1:p1+2*k,[ind1,p1+1:p1+k]) = sparse(k,n-m+k);
        DD([ind1,p1+1:p1+k],p1+k+1:p1+2*k) = sparse(n-m+k,k);
        
        
        DD(ind1, p1+1:p1+k) = Aoff;
        DD(p1+1:p1+k, ind1) = Aoff';    
        
        DD(p1+1:p1+k,p1+1:p1+k) = Bss;
        DD(p1+k+1:p1+2*k,p1+k+1:p1+2*k) = D;
        
        % retrieve indices of the inputs in the residual set
        rind = Order(p1+k+1:p1+2*k);  
        % switch skeleton index to 0 if in residual set
        x(rind, end) = zeros(length(rind), 1);

    end
    PP = [PP; Pis, sparse(n,size(PP,2)-size(Pis,2))];
    Qi = [Qi; Qinvs, sparse(n,size(Qi,2)-size(Qinvs,2))];
    Si = [Si; Sinvs, sparse(n,size(Si,2)-size(Sinvs,2))];
%     % spy DD
%     figure(c+1)
%     spy(DD)
%     xlabel('')
%     % Plot the DOFs
%     figure(c+1)
%     for i = 1:n
%         if x(i,d+1) == 1
%             plot(x(i,1), x(i,2), 'b.', 'MarkerSize', 10)
%             xlim([-2,2])
%             ylim([-2,2])
%             hold on
%         end
%     end
%     hold off
end
Tfactor = toc;
fprintf('Tfactor: %.4e  ',Tfactor)
% %% Reorder DD blocks to compute determinant
% Dperm = speye(n);
% Tempcol = Dperm(:,n/2+1:n/2+k);
% Dperm(:, 2*k+1:n/2+k)=Dperm(:, k+1:n/2);
% Dperm(:, k+1:2*k) = Tempcol;
% DDet = Dperm'*DD*Dperm;
% % spy(DDet)
% dmt = det(full(DDet(1:2*k,1:2*k)));
% DDiag = spdiags(DDet(2*k+1:n,2*k+1:n));
% dmt = dmt*prod(DDiag);
% % Test
% for i = 1:n
%     K(i,:) = f(x(i,:),x(:,:));
% end
% dmtreal = det(K);
% dmterr = norm(dmtreal-dmt);
% fprintf('dmterr: %.4e  ', dmterr)
%% Error analysis -------------------------------------------------

% Test vector
tv = 2*rand(n,1)-1;
tv = tv/norm(tv);
tv1 = tv;
tic
% over levels
for c = 1:Lev
    m = n/2^Lev*2^(c-1);
    %over subdomains
    for z = 1:n/m
        tv1 = PP((c-1)*n+1:c*n, (z-1)*n+1:z*n)*tv1;
        tv1 = Qi((c-1)*n+1:c*n, (z-1)*n+1:z*n)*tv1;
        tv1 = Si((c-1)*n+1:c*n, (z-1)*n+1:z*n)*tv1;
    end
end

tv1 = DD*tv1;

% over levels
for e = Lev:-1:1
    m = n/2^Lev*2^(e-1);
    % over subdomains
    for y = n/m:-1:1
        tv1 = Si((e-1)*n+1:e*n, (y-1)*n+1:y*n)'*tv1;
        tv1 = Qi((e-1)*n+1:e*n, (y-1)*n+1:y*n)'*tv1;
        tv1 = PP((e-1)*n+1:e*n, (y-1)*n+1:y*n)'*tv1;        
    end
end
Tapply = toc;
fprintf('Tapply: %.4e  ',Tapply)
tic
% Apply A to tv for testing
tv2 = zeros(n,1);
for i = 1:n
    if d == 1
        tv2(i) = f(x(i,1:d), x(1:n,1:d))'*tv;
    else
        tv2(i) = f(x(i,1:d), x(1:n,1:d))*tv;
    end
end
tv2 = tv2 + tv;
Testt = toc;
fprintf('Testt: %.2e\n',Testt)
% Spectral norm error estimate
fprintf('L2 error: %.4e\n', norm(tv2 - tv1))
fprintf('n: %d,  Lev: %d,  k: %d\n',n,Lev,k)