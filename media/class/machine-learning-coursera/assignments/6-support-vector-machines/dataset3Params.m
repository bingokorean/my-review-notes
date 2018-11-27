function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

vec = [ 0.01 0.03 0.1 0.3 1 3 10 30 ];
error_vec = zeros(64,3);

%model = svmTrain(X, y, 1, @(x1, x2) gaussianKernel(x1, x2, 1 ));

for c=1:8
    for s=1:8
        model = svmTrain(X, y, vec(c), @(x1, x2) gaussianKernel(x1, x2, vec(s)) );
        pred = svmPredict(model, Xval);
        error_vec(8*(c-1)+s,:) = [ mean(double(pred ~= yval)) vec(c) vec(s) ];
    end
end

min_error = min(error_vec(:,1));
for i=1:64
    if(error_vec(i,1) == min_error)
        C = error_vec(i,2);
        sigma = error_vec(i,3);
    end    
end
% =========================================================================
end