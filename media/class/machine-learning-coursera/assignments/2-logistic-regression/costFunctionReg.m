function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:));
regsum = 0;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Cost Function Reg
H = sigmoid(X*theta);
for i=1:m
    J = J + ( -y(i)*log(H(i)) - (1-y(i))*log(1-H(i)) );
end
J = 1/m*J;
for i=2:n
     regsum = regsum + (theta(i)).^2;
end
J = J + lambda/(2*m)*regsum;

% Gradient for logistic
for j=1:n
    if j==1       
        grad(j) = 1/m*( X(:,j)'*(H - y) );
    else
        grad(j) = 1/m*( X(:,j)'*(H - y) ) + lambda/m*theta(j);
    end
end

% =============================================================

end
