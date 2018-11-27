%% Feature Normalization

function [X_norm, mu, sigma] = featureNormalize(X)
    %FEATURENORMALIZE Normalizes the features in X 
    %   FEATURENORMALIZE(X) returns a normalized version of X where
    %   the mean value of each feature is 0 and the standard deviation
    %   is 1. This is often a good preprocessing step to do when
    %   working with learning algorithms.

    X_norm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));


    % X = [ m n ], mu = [ 1 n ], sigma = [ 1 n ]
    n = length(X(1,:)); % n feature
    %n = 1; % 1¹øÂ° feature¸¸ featureNormalizationÇÏ±â


    for j=1:n
        mu(1,j) = mean(X(:,j));
        sigma(1,j) = std(X(:,j));
    end


    for j=1:n
        X_norm(:,j) = (X(:,j) - mu(1,j)) ./ sigma(1,j);
    end

end
