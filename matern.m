function value = matern(x, y, l, nu)
% evaluate matern kernel with parameters l, nu 
% at point (x,y)
[~,d] = size(x);
if d ==1
    value = 2^(1-nu)/gamma(nu)*...
    (sqrt(2*nu)*abs(x-y)/l).^nu.*...
    besselk(nu, sqrt(2*nu)*abs(x-y)/l);
    value(isnan(value)) = 1;
else
    value = 2^(1-nu)/gamma(nu)*...
    (sqrt(2*nu)*vecnorm(x'-y')/l).^nu.*...
    besselk(nu, sqrt(2*nu)*vecnorm(x'-y')/l);
    value(isnan(value)) = 1;
end
end