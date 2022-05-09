clear;
% sq exp------------------------------------------
l1 = 1;
k1 = @(x,y) sqexp(x,y,l1); 
% exp--------------------------------------------
l2 = 1e+0;
k2 = @(x,y) exponential(x,y,l2); 
% % Matern --------------------------------------
l3 = 1; nu = 5/2;
k3 = @(x,y) matern(x,y,l3,nu);
% % rational quadratic ----------------------------
% l4 = 1; a = 1;
% k4 = @(x,y) rquadratic(x,y,l4,a);
% % biharmonic ---------------------------------- 
% k5 = @(x,y) biharmonic(x,y);
% % dot product----------------------------------
% k = @(x,y) 0 + dot(x,y); 
% % ---------------------------------------------

n = 2^(10); d = 1;
rng(3);
x = 2*rand(n,d)-1;
x = 1*x;
% sort data in k-d tree
x = treesort(x,2^5);
for i = 1:n
    A1(i,:) = k1(x(i,:), x(:,:));
    A2(i,:) = k2(x(i,:), x(:,:));
    A3(i,:) = k3(x(i,:), x(:,:));
end
sval1 = svd(A1);
sval2 = svd(A2);
sval3 = svd(A3);
semilogy(1:n, sval1, 'b', 'LineWidth', 2, 'DisplayName', 'sq. exp.');
hold on
semilogy(1:n, sval2, 'k', 'LineWidth', 2, 'DisplayName', 'exp.');
hold on
semilogy(1:n, sval3, 'r', 'LineWidth', 2, 'DisplayName', 'Matern');
t = sprintf('n=%d,  d=%d\n', n, d);
xlabel(t, 'FontSize', 14)
ylabel('singular value', 'FontSize', 14)
legend('Location', 'northeast', 'FontSize', 14)
legend show
set(gcf, 'Position', [0,0,500,450])
hold off

xlim([0,n+1])
% C = A + eye(n);
% rank(A, 1e-8)
% rank(C(1:n/2, n/2+1: n))
% rank(C(1:n/2, 1:n/2))