function [learned_params, Cost] = nesterov_accelerated_gradient(init_nn_params, alpha, num_iters, ...
    input_layer_size, hidden_layer_size, num_labels, X, y, lambda, gamma)

%m = length(y); % number of training examples
Cost = zeros(num_iters, 1);
theta = init_nn_params;


% First iteration
[J grad] = nnCostFunction(theta, input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda); 
delta = alpha*grad;
theta = theta - delta;


for iter = 2:num_iters
   
    [J N_grad] = nnCostFunction(theta-(gamma*delta), input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda);  
    
    delta = gamma*delta + alpha*N_grad;
    theta = theta - delta;
    
    % Save the cost J in every iteration    
    Cost(iter) = J;   
end

learned_params = theta;
end
