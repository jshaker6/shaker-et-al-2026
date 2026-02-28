function [avgFR, allTrialFR] = calcAvgTrialFR(sp, trialTimes)
% CALCAVGTRIALFR  Compute mean and per-trial firing rates for all clusters.
%   sp         - struct with fields .st (spike times), .clu (cluster IDs), .cids
%   trialTimes - (nTrials x 2) matrix: [start_time, end_time] per trial
%
%   avgFR      - (nClusters x 1) mean firing rate across trials
%   allTrialFR - (nClusters x nTrials) firing rate on each trial

    clu = sp.clu;
    st = sp.st;
    cids = sp.cids;
    nClu = numel(cids);
    numTrials = size(trialTimes, 1);

    stInTrialsLogical = st > trialTimes(:,1)' & st < trialTimes(:,2)';
    cluInTrials = cell(1, numTrials);
    for ii = 1:numTrials
        cluInTrials{ii} = clu(stInTrialsLogical(:, ii));
    end

    avgFR = zeros(nClu, 1);
    allTrialFR = zeros(nClu, numTrials);
    for cidx = 1:nClu
        for trIdx = 1:numTrials
            cluSt = find(cluInTrials{trIdx} == cids(cidx));
            if ~isempty(cluSt)
                allTrialFR(cidx, trIdx) = numel(cluSt) / (trialTimes(trIdx,2) - trialTimes(trIdx,1));
            end
        end
        avgFR(cidx, 1) = mean(allTrialFR(cidx, :), 'omitnan');
    end
end
