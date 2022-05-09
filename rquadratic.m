function value = rquadratic(x,y,l,a)
% evaluate rational quadratic kernel with parameters l, a
[~,d] = size(x);
if d == 1
    value = (1+(abs(x-y).^2)/(2*a*l^2))^(-a);
else
    value = (1+(vecnorm(x'-y').^2)/(2*a*l^2))^(-a);
end 
