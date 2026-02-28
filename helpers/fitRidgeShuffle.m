function [x, varEtest, predB] = fitRidgeShuffle(A, B, trainSet, testSet, inclBins, lambda, useGlmnet)
% FITRIDGESHUFFLE  Fit ridge regression with a preset lambda (no CV search).
%   Used during shuffle iterations where lambda was determined from the
%   original (unshuffled) fit,
%
%   A         - (nTimeBins x nPredictors) predictor matrix
%   B         - (nTimeBins x 1) neural activity vector
%   trainSet  - cell array of logical vectors (K-fold)
%   testSet   - cell array of logical vectors (K-fold)
%   inclBins  - logical vector of bins to include in R2 calculation
%   lambda    - preset regularization parameter(s) from original fit
%   useGlmnet - true if glmnet_matlab is available
%
%   x         - regression coefficients (row 1 = intercept)
%   varEtest  - cross-validated R2
%   predB     - predicted neural activity

    try
        nFolds = numel(trainSet);
        predB = zeros(size(B));
        x = zeros(size(A,2)+1, nFolds);

        for cvFold = 1:nFolds
            Atrain = A(trainSet{cvFold}, :);
            Atest = A(testSet{cvFold}, :);
            Btrain = B(trainSet{cvFold});

            if useGlmnet
                opts.alpha = 0; opts.standardize = true; opts.lambda = lambda(cvFold);
                options = glmnetSet(opts);
                fit = glmnet(Atrain, Btrain, 'gaussian', options);
                x(:, cvFold) = glmnetCoef(fit, lambda(cvFold));
            else
                % Direct ridge regression with preset lambda
                nPred = size(Atrain, 2);
                AtA = Atrain' * Atrain;
                x_body = (AtA + lambda(cvFold) * size(Atrain,1) * eye(nPred)) \ (Atrain' * Btrain);
                intercept = mean(Btrain) - mean(Atrain, 1) * x_body;
                x(:, cvFold) = [intercept; x_body];
            end

            predB(testSet{cvFold}) = [ones(sum(testSet{cvFold}),1) Atest] * x(:, cvFold);
        end

        varEtest = 1 - (sum((B(inclBins) - predB(inclBins)).^2) / ...
                            sum((B(inclBins) - mean(B(inclBins))).^2));
       
    catch
        x = NaN(size(A,2)+1, numel(trainSet));
        predB = NaN(length(B), 1);
        varEtest = NaN;
    end
end
