function TxN = produceTxN(neuronList, sp, startTimes, endTimes, normalizeFlag, spForBLFR)
% PRODUCETXN  Build a trial-by-neuron firing rate matrix from spike data.
%   Counts spikes within per-trial time windows and divides by bin duration
%   to produce firing rates. Optionally normalizes each neuron's rates by
%   its baseline firing rate (counts / (baselineFR + 0.5)).
%
%   neuronList    - (N x 3) matrix: [cidIndex, probeNum, shankNum]
%   sp            - cell array from loadSessionData: sp{pn}(sn) with fields
%                   .st, .clu, .cids, .baselineFR
%   startTimes    - (nTrials x 1) trial window start times (seconds)
%   endTimes      - (nTrials x 1) trial window end times (seconds)
%   normalizeFlag - if true, divide counts by (baselineFR + 0.5)
%   spForBLFR     - sp structure to get baselineFR from (pass sp itself, or
%                   [] if normalizeFlag is false)
%
%   TxN           - (nTrials x nNeurons) firing rate matrix

    nTrials  = numel(startTimes);
    nNeurons = size(neuronList, 1);
    TxN = zeros(nTrials, nNeurons);

    % Build interleaved trial interval edges for histcounts
    startTimes = startTimes(:)';  % ensure row
    endTimes   = endTimes(:)';
    trialIntervals = [startTimes; endTimes];
    trialBinSizes  = endTimes - startTimes;
    trialIntervals = trialIntervals(:);  % interleaved: [s1 e1 s2 e2 ...]

    for nn = 1:nNeurons
        cidIdx = neuronList(nn, 1);
        pn     = neuronList(nn, 2);
        sn     = neuronList(nn, 3);

        cid   = sp{pn}(sn).cids(cidIdx);
        cluSt = sp{pn}(sn).st(sp{pn}(sn).clu == cid);

        [counts, ~] = histcounts(cluSt, trialIntervals(:));
        counts(2:2:end) = [];  % remove inter-trial intervals

        if normalizeFlag && ~isempty(spForBLFR)
            blfr = spForBLFR{pn}(sn).baselineFR(cidIdx);
            counts = counts ./ (blfr + 0.5);
        end

        TxN(:, nn) = (counts ./ trialBinSizes)';  % column vector
    end
end
