function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% h(x) = g(theta' * x) or g(X * theta)
# g(z) = 1/(1+e^-z) or sigmoid(z)
% therefore h(x) = sigmoid(X * theta)

n = size(X, 2);% number of features

J = (1/m * sum(-y .* log(sigmoid(X * theta)) - (1-y) .* log(1 - sigmoid(X * theta)))) + ((lambda / (2 * m)) * sum(theta(2:n) .^ 2));

% X(:, 1) is the first feature x-j, where j = 0.
% This gradient goes to theta-j where j = 0
% no regularization for this theta

% note : the sum of error h(x) - y still iterates through all features + theta
grad(1) = 1/m * X(:, 1)' * (sigmoid(X * theta) - y);

grad(2:n) = (1/m * X(:, 2:n)' * (sigmoid(X * theta)- y)) + (lambda/m * theta(2:n));

% =============================================================

end
