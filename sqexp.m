function value = sqexp(x,y,l)
% evaluate squared exponential kernel with parameter l
[~,d] = size(x);
if d == 1
    value = exp(-(abs(x-y).^2)/(2*l^2));
else
    value = exp(-(vecnorm(x'-y').^2)/(2*l^2));
end 