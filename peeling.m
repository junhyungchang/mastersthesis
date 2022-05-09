clear
% Matrix peeling algorithm for trace
d = 2;
n = 2^14;
k = 40; % target rank
l = 1;
f = @(x,y) sqexp(x,y,l);
smallest = 2^6; % size of leaf level box
Lev = log2(n/smallest); % number of levels
tol = 1e-8;
fprintf('Dim:%d , System size:%d , Target rk:%d , Num levels:%d\n',d, n, k, Lev)
%% Preparing data matrix
rng(5)
x = 4*rand(n,d)-2;

% k-d tree sorting
% number and size of smallest sub-blocks
num = 2^Lev;
x = treesort(x,num);

% %% Generate A (test purposes)
% 
% for i = 1:n
%     A(i,:) = f(x(i, :), x(1:n,:));
% end


%% Begin matrix Peeling
% Initialize low rank factors of A^(l)
U = [];
V = [];
Atil = sparse(2^Lev*k,Lev*k);

%% First level
tic
for c = 1
    % size of sub-blocks
    m = n/2^c;
    %% Generate random matrix
    OmegaNZ = randn(n,k);
    Omega = sparse(n,2*k);
    for j = 1:2^c
        if mod(j,2) == 1
            Omega((j-1)*m+1:j*m, k+1:2*k) = OmegaNZ((j-1)*m+1:j*m,:);
        else
            Omega((j-1)*m+1:j*m, 1:k) = OmegaNZ((j-1)*m+1:j*m,:);
        end
    end
    
    %% Find Y blocks (approximate basis for col space)
    
%     Y = A*Omega;
    for j = 1:n
        if d == 1
            Y(j,:) = f(x(j,:), x(1:n,:))'*Omega;
        else
            Y(j,:) = f(x(j,:), x(1:n,:))*Omega;
        end
    end
        
    %% Orthogonalize Y blocks
    for j = 1:2^c
        if mod(j,2) == 1
            [Q,~] = qr(Y((j-1)*m+1:j*m, 1:k) ,0);
            U((j-1)*m+1:j*m,(c-1)*k+1:c*k) = Q;
        else
            [Q,~] = qr(Y((j-1)*m+1:j*m, k+1:2*k) ,0);
            U((j-1)*m+1:j*m,(c-1)*k+1:c*k) = Q;
        end
    end
    
    %% Form Z matrix
    for j = 1:2^c
        if mod(j,2) == 1
            Omega((j-1)*m+1:j*m, k+1:2*k) = U((j-1)*m+1:j*m,(c-1)*k+1:c*k);
        else
            Omega((j-1)*m+1:j*m, 1:k) = U((j-1)*m+1:j*m,(c-1)*k+1:c*k);
        end
    end

%     Z = A'*Omega;
    for j = 1:n
        if d == 1
            Z(j,:) = f(x(j,:), x(1:n,:))'*Omega;
        else
            Z(j,:) = f(x(j,:), x(1:n,:))*Omega;
        end
    end
    
    %% Take SVD of Z' blocks
    for j = 1:2^c
        if mod(j,2) == 1
            [Uhat, Ahat, Vhat] = svd(Z((j-1)*m+1:j*m, 1:k)', 'econ');
            U(j*m+1:(j+1)*m, (c-1)*k+1:c*k) = U(j*m+1:(j+1)*m, (c-1)*k+1:c*k)*Uhat;
            V(j*m+1:(j+1)*m, (c-1)*k+1:c*k) = Vhat;
            Atil(j*k+1:(j+1)*k, (c-1)*k+1:c*k) = Ahat;
        elseif mod(j,2) == 0
            [Uhat, Ahat, Vhat] = svd(Z((j-1)*m+1:j*m, k+1:2*k)', 'econ');
            U((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k) = U((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k)*Uhat;
            V((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k) = Vhat;
            Atil((j-2)*k+1:(j-1)*k, (c-1)*k+1:c*k) = Ahat;
        end
%         norm(A((2-j)*n/2+1:(3-j)*n/2, (j-1)*n/2+1:j*n/2) -  U((2-j)*m+1:(3-j)*m, (c-1)*k+1:c*k)*Ahat*Vhat')
    end
end

%% Higher levels
for c = 2:Lev
    % size of sub-blocks
    m = n/2^c;
    %% Generate random matrix
    OmegaNZ = randn(n,k);
    Omega = sparse(n,2*k);
    for j = 1:2^c
        if mod(j,2) == 1
            Omega((j-1)*m+1:j*m, k+1:2*k) = OmegaNZ((j-1)*m+1:j*m,:);
        else
            Omega((j-1)*m+1:j*m, 1:k) = OmegaNZ((j-1)*m+1:j*m,:);
        end
    end
    
    %% Find Y blocks (approximate basis for col space)
    
%     Y = A*Omega;
    for j = 1:n
        if d == 1
            Y(j,:) = f(x(j,:), x(1:n,:))'*Omega;
        else
            Y(j,:) = f(x(j,:), x(1:n,:))*Omega;
        end
    end
    
    %% subtract all communication matrices before current level
%     TMat = A;
    for i = 1:c-1
        m1 = n/2^i;
        % Construct UU, VV (low-rank factors of Al)
        UU = sparse(n,2^i*k);
        VV = sparse(2^i*k,n);
        AtilMat = sparse(2^i*k,2^i*k);
        for j = 1:2^i
            % UU
            if mod(j,2) == 1
                UU((j-1)*m1+1:j*m1, j*k+1:(j+1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
            else
                UU((j-1)*m1+1:j*m1, (j-2)*k+1:(j-1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
            end
            % VV
            if mod(j,2) == 1
                VV(j*k+1:(j+1)*k, j*m1+1:(j+1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
            else
                VV((j-2)*k+1:(j-1)*k, (j-2)*m1+1:(j-1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
            end
            % AtilMat
            if mod(j,2) == 1
                AtilMat(j*k+1:(j+1)*k, j*k+1:(j+1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
            else
                AtilMat((j-2)*k+1:(j-1)*k, (j-2)*k+1:(j-1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
            end
        end
        AOmega = (UU*(AtilMat*(VV*Omega)));
        Y = Y - AOmega;
%         TMat = TMat - UU*AtilMat*VV;
%         Indices = find(abs(TMat) < tol);
%         TMat(Indices) = 0;
%         spy(TMat)
    end
         
    %% Orthogonalize Y blocks
    for j = 1:2^c
        if mod(j,2) == 1
            [Q,~] = qr(Y((j-1)*m+1:j*m, 1:k) ,0);
            U((j-1)*m+1:j*m,(c-1)*k+1:c*k) = Q;
        else
            [Q,~] = qr(Y((j-1)*m+1:j*m, k+1:2*k) ,0);
            U((j-1)*m+1:j*m,(c-1)*k+1:c*k) = Q;
        end
    end
    
    %% Form Z matrix
    for j = 1:2^c
        if mod(j,2) == 1
            Omega((j-1)*m+1:j*m, k+1:2*k) = U((j-1)*m+1:j*m,(c-1)*k+1:c*k);
        else
            Omega((j-1)*m+1:j*m, 1:k) = U((j-1)*m+1:j*m,(c-1)*k+1:c*k);
        end
    end

%     Z = A'*Omega;
    for j = 1:n
        if d == 1
            Z(j,:) = f(x(j,:), x(1:n,:))'*Omega;
        else
            Z(j,:) = f(x(j,:), x(1:n,:))*Omega;
        end
    end
    %% subtract all communication matrices before current level
    for i = 1:c-1
        m1 = n/2^i;
        % Construct UU, VV (low-rank factors of Al)
        UU = sparse(n,2^i*k);
        VV = sparse(2^i*k,n);
        AtilMat = sparse(2^i*k,2^i*k);
        for j = 1:2^i
            % UU
            if mod(j,2) == 1
                UU((j-1)*m1+1:j*m1, j*k+1:(j+1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
            else
                UU((j-1)*m1+1:j*m1, (j-2)*k+1:(j-1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
            end
            % VV
            if mod(j,2) == 1
                VV(j*k+1:(j+1)*k, j*m1+1:(j+1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
            else
                VV((j-2)*k+1:(j-1)*k, (j-2)*m1+1:(j-1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
            end
            % AtilMat
            if mod(j,2) == 1
                AtilMat(j*k+1:(j+1)*k, j*k+1:(j+1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
            else
                AtilMat((j-2)*k+1:(j-1)*k, (j-2)*k+1:(j-1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
            end
        end
        AtOmega = (VV'*(AtilMat'*(UU'*Omega)));
        Z = Z - AtOmega;
    end
    
    %% Take SVD of Z' blocks
    for j = 1:2^c
        if mod(j,2) == 1
            [Uhat, Ahat, Vhat] = svd(Z((j-1)*m+1:j*m, 1:k)', 'econ');
            U(j*m+1:(j+1)*m, (c-1)*k+1:c*k) = U(j*m+1:(j+1)*m, (c-1)*k+1:c*k)*Uhat;
            V(j*m+1:(j+1)*m, (c-1)*k+1:c*k) = Vhat;
            Atil(j*k+1:(j+1)*k, (c-1)*k+1:c*k) = Ahat;
        elseif mod(j,2) == 0
            [Uhat, Ahat, Vhat] = svd(Z((j-1)*m+1:j*m, k+1:2*k)', 'econ');
            U((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k) = U((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k)*Uhat;
            V((j-2)*m+1:(j-1)*m, (c-1)*k+1:c*k) = Vhat;
            Atil((j-2)*k+1:(j-1)*k, (c-1)*k+1:c*k) = Ahat;
        end
%         norm(A((2-j)*n/2+1:(3-j)*n/2, (j-1)*n/2+1:j*n/2) -  U((2-j)*m+1:(3-j)*m, (c-1)*k+1:c*k)*Ahat*Vhat')
    end
    
end
tpeel = toc;    
fprintf('tpeel:%.4e  ', tpeel)
%% extracting the trace
tic
Omega = sparse(n, smallest);
numblocks = n/smallest;
for i = 1:numblocks
    Omega((i-1)*smallest+1:i*smallest, :) = speye(smallest);
end
Y = [];
for j = 1:n
    if d == 1
        Y(j,:) = f(x(j,:), x(1:n,:))'*Omega;
    else
        Y(j,:) = f(x(j,:), x(1:n,:))*Omega;
    end
end
%% subtract all communication matrices before current level
%     TMat = A;
for i = 1:Lev
    m1 = n/2^i;
    % Construct UU, VV (low-rank factors of Al)
    UU = sparse(n,2^i*k);
    VV = sparse(2^i*k,n);
    AtilMat = sparse(2^i*k,2^i*k);
    for j = 1:2^i
        % UU
        if mod(j,2) == 1
            UU((j-1)*m1+1:j*m1, j*k+1:(j+1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
        else
            UU((j-1)*m1+1:j*m1, (j-2)*k+1:(j-1)*k) = U((j-1)*m1+1:j*m1,(i-1)*k+1:i*k);
        end
        % VV
        if mod(j,2) == 1
            VV(j*k+1:(j+1)*k, j*m1+1:(j+1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
        else
            VV((j-2)*k+1:(j-1)*k, (j-2)*m1+1:(j-1)*m1) = V((j-1)*m1+1:j*m1,(i-1)*k+1:i*k)';
        end
        % AtilMat
        if mod(j,2) == 1
            AtilMat(j*k+1:(j+1)*k, j*k+1:(j+1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
        else
            AtilMat((j-2)*k+1:(j-1)*k, (j-2)*k+1:(j-1)*k) = Atil((j-1)*k+1:j*k,(i-1)*k+1:i*k);
        end
    end
    AOmega = (UU*(AtilMat*(VV*Omega)));
    Y = Y - AOmega;
%     TMat = TMat - UU*AtilMat*VV;
%     Indices = find(abs(TMat) < tol);
%     TMat(Indices) = 0;
%     spy(TMat)
%     xlabel('')
end
%% compute the trace of leaf blocks, and add up recursively
TracePeel = 0;
for j = 1:numblocks
    TracePeel = TracePeel + trace(Y((j-1)*smallest+1:j*smallest, :)); 
end
textract = toc;
fprintf('textract:%.4e  ', textract)
% compare with exact trace
tic
TraceExact = 0;
for j = 1:n
    TraceExact = TraceExact + f(x(j,:),x(j,:));
end
ttest = toc;
fprintf('ttest:%.4e  ', ttest)
fprintf('error:%.4e\n', TraceExact-TracePeel)