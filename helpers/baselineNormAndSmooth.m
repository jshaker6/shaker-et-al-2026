function [binnedSpNorm, binnedSpNormSmoothed] = baselineNormAndSmooth(sp, binSize, expStartStop, stdev)
% BASELINENORMANDSMOOTH  Bin spikes, baseline-normalize, and apply causal smoothing.
%   Applies a causal (half-Gaussian) smoothing kernel to preserve the temporal
%   relationship between neural activity and behavioral events.
%
%   sp           - struct array (1 x nShanks) with .st, .clu, .cids, .baselineFR
%   binSize      - time bin width in seconds
%   expStartStop - [start, end] times of experiment
%   stdev        - standard deviation of the Gaussian smoothing kernel (seconds)
%
%   binnedSpNorm        - cell array {nShanks x 1}, each (nTimeBins x nClusters)
%   binnedSpNormSmoothed - cell array {nShanks x 1}, each (nTimeBins x nClusters)

    % Construct causal half-Gaussian filter
    samps = ceil((stdev / binSize) * 2 * 3);
    gw = gausswin(samps, 3);
    gw(1:round(numel(gw)/2)) = 0;  % zero the acausal half
    smWin = gw ./ sum(gw);          % normalize

    nShanks = numel(sp);
    binnedSpNorm = cell(nShanks, 1);
    binnedSpNormSmoothed = cell(nShanks, 1);

    for sn = 1:nShanks
        if isempty(sp(sn).cids); continue; end
        [~, binnedSpNorm{sn}] = binSpikesAndBaselineNorm(sp(sn), binSize, expStartStop, sp(sn).baselineFR);
        binnedSpNormSmoothed{sn} = zeros(size(binnedSpNorm{sn}));
        for cidx = 1:size(binnedSpNorm{sn}, 2)
            binnedSpNormSmoothed{sn}(:, cidx) = conv(binnedSpNorm{sn}(:, cidx), smWin, 'same');
        end
    end
end
