%% Gradient Descent Algorithm

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
     %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
     %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
     %   taking num_iters gradient steps with learning rate alpha

     % Initialize some useful values
     m = length(y); % number of training examples

     for iter = 1:num_iters

         n = length(X(1,:)); % n features

         temptheta = zeros(n, 1);
         H = X*theta;

         for j=1:n  
             sum = X(:,j)'*( H - y );            
             temptheta(j) = theta(j) - alpha/m*sum;
         end

         % simultaneously update theta
         for j=1:n
             theta(j) = temptheta(j);
         end

         % Save the cost J in every iteration    
         J_history(iter) = computeCostMulti(X, y, theta);

     end
end
