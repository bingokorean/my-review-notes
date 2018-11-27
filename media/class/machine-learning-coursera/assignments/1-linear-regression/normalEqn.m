%% Normal Equation

function [theta] = normalEqn(X, y)
    %NORMALEQN Computes the closed-form solution to linear regression 
    %   NORMALEQN(X,y) computes the closed-form solution to linear 
    %   regression using the normal equations.

    theta = pinv((X'*X))*X'*y; % Just one-shot!!! (it's closed-form)
end
