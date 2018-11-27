function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost Function
for i=1:m
    J = J + (X(i,:)*theta - y(i))^2;
end

J = 1/(2*m) * J;

n = length(theta(:,1));
R = 0;
for j = 2:n
    R = R + (theta(j))^2;
end

R = lambda / (2*m) * R;
J = J + R;


% Gradient
tempTheta = theta;
for i=1:length(tempTheta(1,:))  % theta 행렬 잘 확인한후 0을 대입할 것.
    tempTheta(1,i) = 0;
end

grad = 1/m*(X'*(X*theta - y)) + lambda/m*tempTheta;

% =========================================================================
grad = grad(:);
end