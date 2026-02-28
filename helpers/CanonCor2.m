% [a b R2 V] = CanonCor2(Y, X)
%
% Reduced rank regression via canonical correlation analysis. Finds the
% linear combinations of X that predict the largest variance fractions of Y.
%
% Y is the dependent variable (neurons x timepoints), X is the independent
% variable (predictor matrix).
%
% R2 is the fraction of total variance of Y explained by the nth projection.
%
% The approximation of Y based on the first n projections is:
%   Y = X * b(:,1:n) * a(:,1:n)';
%
% V is the value of the nth linear combination of X.

function [a, b, R2, V] = CanonCor2(Y, X)

XSize = size(X, 2);

BigCov = cov([X, Y]);
CXX = BigCov(1:XSize, 1:XSize);
CYX = BigCov(XSize+1:end, 1:XSize);

eps = 1e-7;
CXX = CXX + eps*eye(XSize);

CXXMH = CXX ^ -0.5;

M = CYX * CXXMH;

[d, s, c] = svd(M, 0);

b = CXXMH * c;
a = d * s;

R2 = (diag(s).^2) / sum(var(Y));

if (nargout > 3)
    V = X * b;
end
