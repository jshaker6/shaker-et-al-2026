function [binnedSp, binnedSpNorm] = binSpikesAndBaselineNorm(sp, binSize, expStartStop, baselineFR)
% BINSPIKESANDBASELINENORM  Bin spike trains and normalize by baseline firing rate.
%   sp           - struct with .st, .clu, .cids
%   binSize      - time bin width in seconds
%   expStartStop - [start, end] times of experiment
%   baselineFR   - (nClusters x 1) baseline firing rates for normalization
%
%   binnedSp     - (nTimeBins x nClusters) binned firing rates (Hz)
%   binnedSpNorm - (nTimeBins x nClusters) baseline-normalized firing rates

    tt = expStartStop(1):binSize:expStartStop(2);
    nT = numel(tt) - 1;

    clu = sp.clu;
    st = sp.st;
    cids = sp.cids;
    nClu = numel(cids);

    binnedSp = zeros(nT, nClu);
    binnedSpNorm = zeros(nT, nClu);

    for cidx = 1:nClu
        cluSt = st(clu == cids(cidx));
        binnedSp(:, cidx) = histcounts(cluSt, tt)' / binSize;
    end
    for cidx = 1:nClu
        binnedSpNorm(:, cidx) = binnedSp(:, cidx) / (baselineFR(cidx, 1) + 1);
    end
end
