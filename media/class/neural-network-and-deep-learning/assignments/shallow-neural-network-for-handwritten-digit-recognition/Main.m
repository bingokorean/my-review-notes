%% Implementation of shallow neural network


%clear ; close all; clc

%% 1. Load & Visualize Data

images = loadMNISTImages('train-images.idx3-ubyte');   % [ 784 x 60000 ] 
labels = loadMNISTLabels('train-labels.idx1-ubyte');   % [ 60000 x 1 ]

%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));
images = transpose(images);   % [ 60000 x 784 ] 
X = images;
y = labels;

test_X = loadMNISTImages('t10k-images.idx3-ubyte');   
test_y = loadMNISTLabels('t10k-labels.idx1-ubyte'); 
test_X = transpose(test_X);

%fprintf('1. Load & Visualize Data FINISH!!!. Press enter to continue.\n');
%pause;


%% 2. Initialize parameters

input_layer_size  = 784;  % 28 x 28 Input Images of Digits
hidden_layer_size = 50;  % 100 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (mapped "0" to label 10)   

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 1;
alpha = 0.5;
gamma = 0.9;
num_iters = 1000;
options = optimset('MaxIter', num_iters);

%fprintf('2. Initialize parameters FINISH!!!. Press enter to continue.\n');
%pause;


%% 3. Train shallow neural network
%fprintf('\nTraining Neural Network... \n')

tic;
% fmincg library 
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% 1. fmincg Library
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%fprintf('fmincg   ');

% 2. Nesterov accelerated gradient
%[nn_params, cost] = nesterov_accelerated_gradient(initial_nn_params, alpha, num_iters, ...
%    input_layer_size, hidden_layer_size, num_labels, X, y, lambda, gamma);
%fprintf('Nesterov   ');

% 3. Simple gradient
%[nn_params, cost] = simple_gradient(initial_nn_params, alpha, num_iters, ...
%    input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
%fprintf('Simple   ');

% 4. Momentum gradient
[nn_params, cost] = momentum_gradient(initial_nn_params, alpha, num_iters, ...
    input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Momentum   ');
toc


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%fprintf('3. Train shallow neural network FINISH!!!. Press enter to continue.\n');
%pause;


%% 4. Performance
pred = predict(Theta1, Theta2, test_X);
fprintf('Test Set Accuracy: %f', mean(double(pred == test_y)) * 100);


fprintf('   lambda: %f', lambda);
fprintf('   alpha: %f', alpha);
fprintf('   gamma: %f', gamma);
fprintf('   num_iters: %f \n', num_iters);

