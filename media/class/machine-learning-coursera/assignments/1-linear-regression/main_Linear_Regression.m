%% Linear Regression (Predicting house prices)
% # Linear regression with on variable 
% # Linear regression with multiple variables

%% 1. Linear regression with one variable 
%% 1.1 Ploting Data 
clear ; close all; clc % Starting point
data1 = load('ex1data1.txt');
fprintf('***** Show just a single data of one variable\n');
data1(1,:) % Show a single data
X1 = data1(:, 1); y1 = data1(:, 2);
m1 = length(y1); % number of training examples
plotData(X1, y1);

%% 1.2 Gradient Descent
X1 = [ones(m1, 1), data1(:,1)]; % Add a column of ones to x
theta1 = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% compute and display initial cost
computeCost(X1, y1, theta1);

% run gradient descent
theta1 = gradientDescent(X1, y1, theta1, alpha, iterations);
% after gradient algorithm, now we have a optimal paramter, theta

% print theta to screen
fprintf('***** Theta found by gradient descent\n');
fprintf(' - theta(1): %f\n - theta(2): %f \n', theta1(1), theta1(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X1(:,2), X1*theta1, '-') % Simple line based on the optimal parameters
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta1;
fprintf('***** For population = 35,000, we predict a profit of %f \n',...
    predict1*10000);
predict2 = [1, 7] * theta1;
fprintf('***** For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% 1.3 Visualizing J 
% The purpose of these graphs is to show how $J(\theta)$ varies with
% changes in $\theta(0)$ and $\theta(1)$. The cost function $J(\theta)$ is
% bowl-shaped and so has a global minimum. This is easier to see in
% the contour plot than in the 3D surface plot. This minimum is the optimal
% point for $\theta(0)$ and $\theta(1)$, and each step of gradient descent
% moves closer to this point.

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X1, y1, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta1(1), theta1(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%% 2. Linear regression with multiple variable
% In this part, we'll implement linear regression with multiple variables
% to predict the house prices. The dataset contains a training set of
% housing prices in Portland, Oregon. The first column indicates the size of the
% house, the second column indicates the number of bedrooms, and the third
% column indicates the house price.

%% 2.1 Load Data
data2 = load('ex1data2.txt');
fprintf('***** Show Data with multiple variable\n');
data2(1,:) % Show the data with one variable
X2 = data2(:, 1:2);
y2 = data2(:, 3);
m = length(y2);
%% 2.2 Feature Normalization
% Do feature normalization before adding intercept
% By looking at the values from dataset, note that house sizes are about
% 1000 times the number of bedrooms. When features differ by orders of
% magnitude, performing feature scaling can make gradient descent converge
% much more quickly.

[X2, mu, sigma] = featureNormalize(X2); 

% Add intercept term to X
X2 = [ones(m, 1) X2];

%% 2.3 Gradient Descent 
% Below picture: If we pick a learning rate within a good range (i.e.,
% ($\alpha=0.01$), our plot look similar to *Blue Line*. If your graph
% looks very different with this, espeicailly if your value of $J(\theta)$
% increases or even blows up, then you should adjust your learning rate,
% $\alpha$. We recommend trying to vary the value of the learning rate $\alpha$ on a
% log-scale at multiplicative steps of about 3 times the previous value
% (i.e., 0.3, 0.1(Blcak Line), 0.03(Red Line), 0.01(Blue Line) etc.)
% With a small learning rate, you should find that gradient descent takes a
% very long time to converge to the optimal value. Conversely, with a large
% learning rate, gradient descent might not converge or might even diverge.

% Choose some alpha value
alpha = 0.01; alpha2 = 0.03; alpha3 = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1); theta2 = zeros(3, 1); theta3 = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X2, y2, theta, alpha, num_iters);
[theta2, J_history2] = gradientDescentMulti(X2, y2, theta2, alpha2, num_iters);
[theta3, J_history3] = gradientDescentMulti(X2, y2, theta3, alpha3, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
hold on;
plot(1:numel(J_history2), J_history2, '-r', 'LineWidth', 2);
plot(1:numel(J_history3), J_history3, '-k', 'LineWidth', 2);

xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('***** Theta computed from gradient descent\n');
fprintf('***** theta [3 x 1] \n');
fprintf('    %f \n', theta);

% Estimate the price of a 1650 sq-ft, 3 br house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = theta(1)+((1650-mu(1))/sigma(1)*theta(2))+((3-mu(2))/sigma(2)*theta(3)); % You should change this

fprintf(['***** Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent)\n']);
fprintf(' - price: %f \n', price);    
%% 2.4 Normal Equations 

% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y); % there is no need alpha and the number of iteration

% Display normal equation's result
fprintf('***** Theta computed from the normal equations:\n');
fprintf('***** theta [3 x 1] \n');
fprintf('    %f \n', theta);
% Estimate the price of a 1650 sq-ft, 3 br house
price = theta(1)+1650*theta(2)+3*theta(3); % You should change this
fprintf(['***** Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations)\n']);
fprintf(' - price: %f \n', price);
