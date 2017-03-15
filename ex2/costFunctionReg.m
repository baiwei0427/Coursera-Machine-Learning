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
%
% Assume that we have m training examples and n + 1 features.
% theta: (n + 1) * 1
% X: m * (n + 1)
% y: m * 1
z = X * theta; %m * 1
h = sigmoid(z); %m * 1
logisf = (-y).*log(h) - (1-y).*log(1-h); %m * 1
theta(1) = 0;
J = sum(logisf) / m + sum(theta.^2) * lambda / 2 / m; 
grad = (X' * (h - y)) / m + theta * lambda / m; 
% =============================================================

end
