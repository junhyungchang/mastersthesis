function [P, T]  = IDQR(A, k)
% Compute ID via Column pivoted QR factorization of a tall matrix.
% the matrix entries are generated within the function
% x is input data
% i1 is index set for rows
% i2 is index set for columns
% f is kernel evaluation function
% k is target rank plus oversampling: k ~ 15

[~,n] = size(A);
[~,R,P] = qr(A,0);
R1 = R(:, 1:k);
R2 = R(:, k+1:n);
T = R1\R2;
