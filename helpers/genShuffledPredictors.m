function [AstimOutsideLShuff, AstimOutsideRShuff, AstimCenterShuff, ...
          AstimLatentShuff, AchoiceShuff, AlatentMoveShuff] = ...
    genShuffledPredictors(blockTL, inclTrials, binSize, expStartStop, kernelLags, pseudolatent)
% GENSHUFFLEDPREDICTORS  Construct shuffled predictor matrices for significance testing.
%   Each task variable is shuffled independently while preserving key statistical
%   structure, enabling assessment of each variable's unique contribution to
%   neural encoding via comparison to the full (unshuffled) model.
%
%   Three distinct shuffling strategies are used:
%
%   1. STIMULUS POSITION: Stimulus positions are permuted within each block,
%      preserving the block-level distribution of trial types while breaking
%      the trial-by-trial stimulus-response mapping.
%
%   2. CHOICE: Response labels are shuffled using an autocorrelation-preserving
%      algorithm (shuffle_sequences) that maintains the run-length distribution
%      of consecutive same-choice trials.
%
%   3. LATENT CONTEXT: Replaced with pseudosession latent traces generated from
%      the fitted context belief model run on simulated task sessions with the
%      same block structure. This preserves the temporal dynamics of the latent variable 
%      while breaking the precise timing of its transitions in the real session.
%
%   blockTL      - behavioral data struct
%   inclTrials   - logical vector of included trials
%   binSize      - time bin width in seconds
%   expStartStop - [start, end] times of experiment
%   kernelLags   - {2 x 1} cell: stim and move kernel lag indices
%   pseudolatent - (nTrials x 1) latent values from one pseudosession

    tt = expStartStop(1):binSize:expStartStop(2);
    nT = numel(tt) - 1;

    nK1 = numel(kernelLags{1});
    nK2 = numel(kernelLags{2});

    numTrials = sum(inclTrials);
    blockType = blockTL.blockType(inclTrials);
    stimPos = blockTL.stimPosition(inclTrials);
    response = blockTL.responseValues(inclTrials);
    pseudolatent = rescale(pseudolatent, -1, 1);

    stimOn = blockTL.stimulusOnTimes(inclTrials) - expStartStop(1);
    moveStart = blockTL.moveTimes(inclTrials, 1) - expStartStop(1);

    stimBinsAll = round(stimOn / binSize);
    moveBinsAll = round(moveStart / binSize);

    % =====================================================================
    % 1. SHUFFLED STIMULUS POSITIONS (within-block permutation)
    % =====================================================================
    % Permute stimPosition labels independently within each block, preserving
    % the number of each trial type per block while breaking the trial-by-trial
    % stimulus identity.
    leftBlockTrials = find(blockType == -1);
    rightBlockTrials = find(blockType == 1);
    stimPosShuff = stimPos;
    stimPosShuff(leftBlockTrials) = stimPos(leftBlockTrials(randperm(numel(leftBlockTrials))));
    stimPosShuff(rightBlockTrials) = stimPos(rightBlockTrials(randperm(numel(rightBlockTrials))));

    % Shuffled trial types
    isOutsideLShuff  = (blockType == -1) & (stimPosShuff == -1);
    isCenterLBShuff  = (blockType == -1) & (stimPosShuff ==  1);
    isCenterRBShuff  = (blockType ==  1) & (stimPosShuff == -1);
    isOutsideRShuff  = (blockType ==  1) & (stimPosShuff ==  1);

    % Stimulus times and movement cutoffs follow the SHUFFLED trial identities.
    % This means the stim onset time for a "shuffled outside left" trial is
    % the actual stim time of whichever trial received that shuffled label.
    AstimOutsideLShuff = zeros(nT, nK1);
    AstimOutsideRShuff = zeros(nT, nK1);
    AstimCenterShuff   = zeros(nT, nK1);

    trialIdx = find(isOutsideLShuff);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimOutsideLShuff(binPos, kidx) = 1;
            end
        end
    end

    trialIdx = find(isOutsideRShuff);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimOutsideRShuff(binPos, kidx) = 1;
            end
        end
    end

    trialIdx = find(isCenterLBShuff | isCenterRBShuff);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimCenterShuff(binPos, kidx) = 1;
            end
        end
    end

    % =====================================================================
    % 2. SHUFFLED LATENT CONTEXT (from pseudosession)
    % =====================================================================
    % The pseudosession latent trace comes from a simulated behavioral session
    % with the same block structure, run through the fitted context belief model.
    % We assign these pseudolatent values to the REAL trial types to construct
    % the shuffled context predictors.

    % Stim-aligned latent predictor: uses pseudolatent at center stim trials
    % (split by real trial type)
    isCenterLB = (blockType == -1) & (stimPos == 1);
    isCenterRB = (blockType ==  1) & (stimPos == -1);

    AstimLatentShuff = zeros(nT, nK1);
    for ttype = {find(isCenterLB)', find(isCenterRB)'}
        trialIdx = ttype{1};
        for jj = 1:numel(trialIdx)
            t = trialIdx(jj);
            for kidx = 1:nK1
                binPos = stimBinsAll(t) + kernelLags{1}(kidx);
                if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                    AstimLatentShuff(binPos, kidx) = pseudolatent(t);
                end
            end
        end
    end

    % Movement-aligned latent predictor: uses pseudolatent on all trials
    AlatentMoveShuff = zeros(nT, nK2);
    for jj = 1:numTrials
        for kidx = 1:nK2
            binPos = moveBinsAll(jj) + kernelLags{2}(kidx);
            if binPos >= 1 && binPos <= nT && binPos >= stimBinsAll(jj)
                AlatentMoveShuff(binPos, kidx) = pseudolatent(jj);
            end
        end
    end

    % =====================================================================
    % 3. SHUFFLED CHOICE (autocorrelation-preserving)
    % =====================================================================
    % Response labels are shuffled using shuffle_sequences, which preserves
    % the run-length distribution of consecutive same-choice trials.
    responseShuff = shuffle_sequences(response);

    AchoiceShuff = zeros(nT, nK2);
    for jj = 1:numTrials
        for kidx = 1:nK2
            binPos = moveBinsAll(jj) + kernelLags{2}(kidx);
            if binPos >= 1 && binPos <= nT && binPos >= stimBinsAll(jj)
                AchoiceShuff(binPos, kidx) = responseShuff(jj);
            end
        end
    end
end
