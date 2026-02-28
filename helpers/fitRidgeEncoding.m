function [x, varEtest, lambda, predB] = fitRidgeEncoding(A, B, trainSet, testSet, inclBins, useGlmnet)
% FITRIDGEENCODING  Fit ridge regression with cross-validated lambda selection.
%   Uses glmnet if available (recommended), otherwise falls back to MATLAB's
%   built-in lasso() with near-zero Alpha as an approximate ridge (hasn't been tested thoroughly).
%
%   A         - (nTimeBins x nPredictors) predictor matrix
%   B         - (nTimeBins x 1) neural activity vector
%   trainSet  - cell array of logical vectors (K-fold)
%   testSet   - cell array of logical vectors (K-fold)
%   inclBins  - logical vector of bins to include in R2 calculation
%   useGlmnet - true if glmnet_matlab is available
%
%   x         - (nPredictors+1 x nFolds) regression coefficients (row 1 = intercept)
%   varEtest  - scalar cross-validated R2 (coefficient of determination)
%   lambda    - selected regularization parameter(s)
%   predB     - (nTimeBins x 1) predicted neural activity

    try
        nFolds = numel(trainSet);
        predB = zeros(size(B));
        x = zeros(size(A,2)+1, nFolds);
        lambda = zeros(nFolds, 1);

        for cvFold = 1:nFolds
            Atrain = A(trainSet{cvFold}, :);
            Atest = A(testSet{cvFold}, :);
            Btrain = B(trainSet{cvFold});

            if useGlmnet
                opts.alpha = 0; opts.standardize = true;
                options = glmnetSet(opts);
                fit = cvglmnet(Atrain, Btrain, 'gaussian', options, [], 10, [], true);
                x(:, cvFold) = cvglmnetCoef(fit, 'lambda_min');
                lambda(cvFold) = fit.lambda_min;
            else
                % Note: this option is not thoroughly tested
                [B_lasso, fitInfo] = lasso(Atrain, Btrain, 'Alpha', 1e-4, 'CV', 5);
                idxMin = fitInfo.IndexMinMSE;
                x(:, cvFold) = [fitInfo.Intercept(idxMin); B_lasso(:, idxMin)];
                lambda(cvFold) = fitInfo.Lambda(idxMin);
            end

            predB(testSet{cvFold}) = [ones(sum(testSet{cvFold}),1) Atest] * x(:, cvFold);
        end

        varEtest = 1 - (sum((B(inclBins) - predB(inclBins)).^2) / ...
                        sum((B(inclBins) - mean(B(inclBins))).^2));

    catch
        x = NaN(size(A,2)+1, numel(trainSet));

        predB = NaN(length(B), 1);
        varEtest = NaN;
        lambda = NaN;
    end
end
