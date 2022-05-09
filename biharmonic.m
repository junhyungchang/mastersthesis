function value = biharmonic(x,y)
% evaluate biharmonic kernel with parameters l, a
[~,d] = size(x);
if d == 1
    value = norm(x-y).^2*log(norm(x-y));
else
    value = vecnorm(x-y).^2*log(vecnorm(x-y));
end 
