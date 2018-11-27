function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
% X = [ 5000 400 ]
a1 = X;
a1 = [ones(m, 1) a1];
% a1 = [5000 401 ]

for i=1:m
    a2(i,:) = sigmoid( a1(i,:) * Theta1' );    
end

a2 = [ones(m, 1) a2];
% a2 = [ 5000 26 ]

for i=1:m
    a3(i,:) = sigmoid( a2(i,:) * Theta2' );
end
% a3 = [ 5000 10 ]

a3(1,:)
max_value = max( a3(1,:), [], 2 )

for i=1:m
    max_value = max( a3(i,:), [], 2 );
    for j=1:num_labels       
        if( a3(i,j) == max_value )
            p(i) = j;
        end
    end
end
% =========================================================================
end
