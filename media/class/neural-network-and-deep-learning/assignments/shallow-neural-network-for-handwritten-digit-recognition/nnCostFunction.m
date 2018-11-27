function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

                               
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
             
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

tempTheta2 = Theta2;             
             
             
% Setup some useful variables
m = size(X, 1);
         
J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));


% --------------------------- Part 1 --------------------------- %

% H를 만드는 과정
a1 = X; % a1 = [ 60000 x 784 ]
a1 = [ones(m,1) a1 ];   % add a0(1),    a1 = [ 60000 x 784+1 ]
z2 = a1 * Theta1';  % z2 = [ 60000 x 100 ]
a2 = sigmoid(z2);
a2 = [ones(m,1) a2 ];   % add a0(2),    a2 = [ 60000 x 100+1 ]
z3 = a2 * Theta2';  % z3 = [ 60000 x 10 ]
H = sigmoid(z3);    % H = [ 60000 x 10 ]

% y를 10개의 라벨로 펼치는 과정
Y = zeros(m,num_labels);
for i=1:m
    for k=1:num_labels
        if(y(i)==k)
            Y(i,k) = 1;   % Y = [ 60000 x 10(num_labels) ]
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

a3 = H;     % a3 = [ 60000 x 10 ]
d3 = a3 - Y;    % d3 = [ 60000 x 10 ]
Theta2 = Theta2(:,2:end);   % Theta2 = [ 10 x 100+1 ] -change-> [ 10 x 100 ]
d2 = d3 * Theta2 .* sigmoidGradient(z2);    % d2 = [ 60000 x 100 ]

D1 = d2' * a1;      % D1 = [ 100 x 784 ]
D2 = d3' * a2;      % D2 = [ 10 X 100+1 ]

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

Theta1_grad = Theta1_grad + R1;     % Theta1_grad = [ 100 x 784+1 ]
Theta2_grad = Theta2_grad + R2;     % Theta2_grad = [ 10 x 100+1 ]



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
