function value = exponential(x,y,l)
% evaluate exponential kernel with parameter l
[~,d] = size(x);
if d == 1
    value = exp(-(abs(x-y))/l);
else
    value = exp(-(vecnorm(x'-y'))/l);
end 