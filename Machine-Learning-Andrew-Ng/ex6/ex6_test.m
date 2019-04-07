load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = C;
[C2, sigma2] = meshgrid(C, sigma);
csig = [C2(:), sigma2(:)];
csignum = size(csig, 1);

errors = zeros(csignum, 1);

for i = 1:size(csig, 1)
  C = csig(i, 1);
  sigma = csig(i, 2);
  % Train the SVM
  fprintf('C = %f, sigma = %f training', C, sigma);
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  predictions = svmPredict(model, Xval);
  % ~= is not equals. get vector of mismatches and mean of total
  errors(i) = mean(double(predictions ~= yval));  
  %visualizeBoundary(X, y, model);
  %fprintf('C = %f, sigma = %f boundary', C, sigma);
  %pause;
end

% find index of minimum error - this idnex of C and sigma should be used
[val, ind] = min(errors);
% got index no. 35