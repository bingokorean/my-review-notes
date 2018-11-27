%% Cost Function with Multiple variables

function J = computeCostMulti(X, y, theta)
    %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    %   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y

    % Initialize some useful values
    m = length(y); % number of training examples

    J = 1/(2*m)*(X*theta-y)'*(X*theta-y);

    % ***** In case of only one variable (= one feature)
    % H = X*theta; % because of X is a matrix, their position is switched.
    % for i=1:m
    %     J = J + (H(i)-y(i))^2;
    % end
    % J = 1/(2*m)*J;
end
