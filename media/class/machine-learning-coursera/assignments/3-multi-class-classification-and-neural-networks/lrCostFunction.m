function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Cost Function과 gradient 계산할때는 x0=1를 추가해줄 필요없다. cost함수는 불러져서 사용되어지는 거기때문에,
% 불러질때 x에는 x0가 이미 추가되어 있을 것이다.

% Cost Function
H = sigmoid(X * theta);
J = 1/m*(( -y' * log(H) ) - ( (1-y') * log(1-H) ));
J = J + lambda/(2*m)*(sum(theta(2:end).^2));    % regularization  % exempt for theta(0) (여기서는 theta(0)가 theta(1)이다.)

% Gradient Descent
grad = 1/m*( X'*(H - y) );
temp = theta;
temp(1) = 0;
grad = grad + lambda/m*temp;

% =============================================================

grad = grad(:);

end
