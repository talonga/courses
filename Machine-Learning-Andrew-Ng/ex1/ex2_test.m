clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, 0.1);

theta = zeros(3, 1);
[theta, J_history2] = gradientDescentMulti(X, y, theta, alpha, num_iters, 1);

theta = zeros(3, 1);
[theta, J_history3] = gradientDescentMulti(X, y, theta, alpha, num_iters, 10);

theta = zeros(3, 1);
[theta, J_history4] = gradientDescentMulti(X, y, theta, alpha, num_iters, 100);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

hold on;
plot(1:numel(J_history), J_history2, 'r', 'LineWidth', 2);
plot(1:numel(J_history), J_history3, 'g', 'LineWidth', 2);
plot(1:numel(J_history), J_history4, 'y', 'LineWidth', 2);