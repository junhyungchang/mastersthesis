clear
% hyperparameter MLE via gradient fminunc

%% Setup ----------------------------------------------------------

% target rank
r = 20;
% oversampling parameter
p = 5; 
x0 = 1;
options = optimoptions('fminunc','Display','iter','SpecifyObjectiveGradient',true);
[x,fval,exitflag,output] = fminunc(@loglhood,x0,options);

