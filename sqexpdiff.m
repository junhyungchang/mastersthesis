function value = sqexpdiff(x,y,l)
% derivative of RBF kernel wrt l

[~,d] = size(x);
if d == 1
    value = abs(x-y).^2.*sqexp(x,y,l)/l^3;
else
    value = vecnorm(x'-y').^2.*sqexp(x,y,l)/l^3;
end 
