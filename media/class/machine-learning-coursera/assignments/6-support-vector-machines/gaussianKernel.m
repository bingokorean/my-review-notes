function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);     % row vector를 column vector로 변환

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

sim = exp( -( ((x1-x2)'*(x1-x2)) / (2*(sigma^2) )) );   % 괄호 꼼꼼하게 해줄 것!

%n = length(x1); % x1의 row, column중 제일 긴 size 반환
%sum = 0;
%for i=1:n
%    sum = sum + ( x1(i) - x2(i) )^2;
%end

%sim = exp( -sum/2*(sigma^2) ); 
% =============================================================
    
end
