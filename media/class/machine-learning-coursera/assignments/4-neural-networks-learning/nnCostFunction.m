function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
             
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

tempTheta2 = Theta2;             
             
             
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% --------------------------- Part 1 --------------------------- %

% H를 만드는 과정
a1 = X; % a1 = [ 5000 x 400 ]
a1 = [ones(m,1) a1 ];   % add a0(1),    a1 = [ 5000 x 401 ]
z2 = a1 * Theta1';  % z2 = [ 5000 x 25 ]
a2 = sigmoid(z2);
a2 = [ones(m,1) a2 ];   % add a0(2),    a2 = [ 5000 x 26 ]
z3 = a2 * Theta2';  % z3 = [ 5000 x 10 ]
H = sigmoid(z3);    % H = [ 5000 x 10 ]

% y를 10개의 라벨로 펼치는 과정
Y = zeros(m,num_labels);
for i=1:m
    for k=1:num_labels
        if(y(i)==k)
            Y(i,k) = 1;   % Y = [ 5000 x 10(num_labels) ]
        end
    end
end

% NN Cost Function
for i=1:m
    for k=1:num_labels
        J = J + (-Y(i,k)*log(H(i,k)) - ((1-Y(i,k))*log(1-H(i,k))));
    end
end
J = 1/m * J;

% Add Regularization to Cost Function
R = 0;
for j=1:hidden_layer_size
    for k=2:input_layer_size+1
        R = R + (Theta1(j,k)).^2;
    end
end

for j=1:num_labels
    for k=2:hidden_layer_size+1
        R = R + (Theta2(j,k)).^2;
    end
end

R = lambda/(2*m) * R;

J = J + R;


% --------------------------- Part 2 --------------------------- %

a3 = H;     % a3 = [ 5000 x 10 ]
d3 = a3 - Y;    % d3 = [ 5000 x 10 ]
Theta2 = Theta2(:,2:end);   % Theta2 = [ 10 x 26 ] -change-> [ 10 x 25 ]
d2 = d3 * Theta2 .* sigmoidGradient(z2);    % d2 = [ 5000 x 25 ]

D1 = d2' * a1;      % D1 = [ 25 x 400 ]
D2 = d3' * a2;      % D2 = [ 10 X 26 ]

D1 = 1/m * D1;
D2 = 1/m * D2;

Theta1_grad = D1;
Theta2_grad = D2;

% --------------------------- Part 3 --------------------------- %

for i=1:length(Theta1(:,1))
    Theta1(i,1) = 0;
end

R1 = lambda/m * Theta1;

for i=1:length(tempTheta2(:,1))
    tempTheta2(i,1) = 0;
end

R2 = lambda/m * tempTheta2;

Theta1_grad = Theta1_grad + R1;     % Theta1_grad = [ 25 x 401 ]
Theta2_grad = Theta2_grad + R2;     % Theta2_grad = [ 10 x 26 ]



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
