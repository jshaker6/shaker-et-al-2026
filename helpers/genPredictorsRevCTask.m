function [AstimOutsideL, AstimOutsideR, AstimCenter, AstimLatent, Amovement, Achoice, AlatentMove] = ...
    genPredictorsRevCTask(blockTL, inclTrials, binSize, expStartStop, kernelLags)
% GENPREDICTORSREVCTASK  Construct task event predictor matrices for kernel regression.
%   Builds time-lagged predictor matrices for each task variable, encoding
%   the timing and identity of visual stimuli (aligned to stim onset) and
%   motor/decision variables (aligned to movement onset). Visual stimulus
%   kernels are truncated at movement onset, and movement kernels are
%   truncated at stimulus onset, preventing temporal overlap.
%
%   PREDICTORS:
%     AstimOutsideL - Outside left stimulus kernel (Left Block, stimPosition=-1)
%     AstimOutsideR - Outside right stimulus kernel (Right Block, stimPosition=+1)
%     AstimCenter   - Center stimulus kernel (common component across blocks)
%     AstimLatent   - Center stimulus kernel modulated by latent context belief
%     Amovement     - Movement kernel (all trials, unsigned)
%     Achoice       - Choice kernel (movement-aligned, signed by response direction)
%     AlatentMove   - Movement kernel modulated by latent context belief(movement-aligned, scaled by latent belief)
%
%   The 8th predictor (context-choice interaction = Achoice .* AlatentMove) is
%   computed in the main script.
%
%   blockTL      - behavioral data struct from FigShare blockTL.mat
%   inclTrials   - logical vector of included trials
%   binSize      - time bin width in seconds
%   expStartStop - [start, end] times of experiment
%   kernelLags   - {2 x 1} cell: {1} = stim lag indices, {2} = move lag indices

    tt = expStartStop(1):binSize:expStartStop(2);
    nT = numel(tt) - 1;

    nK1 = numel(kernelLags{1});  % number of stim kernel time lags
    nK2 = numel(kernelLags{2});  % number of move kernel time lags

    numTrials = sum(inclTrials);
    blockType = blockTL.blockType(inclTrials);
    stimPos = blockTL.stimPosition(inclTrials);
    response = blockTL.responseValues(inclTrials);
    latent = rescale(blockTL.latent(inclTrials), -1, 1);

    % Stimulus and movement times relative to experiment start
    stimOn = blockTL.stimulusOnTimes(inclTrials) - expStartStop(1);
    moveStart = blockTL.moveTimes(inclTrials, 1) - expStartStop(1);

    % --- Define trial types ---
    % Four stimulus conditions based on block identity and stimulus position:
    %   Outside Left:    Left Block (blockType=-1), stimPosition=-1 (left screen)
    %   Center (Left B): Left Block (blockType=-1), stimPosition=+1 (center screen)
    %   Center (Right B):Right Block (blockType=+1), stimPosition=-1 (center screen)
    %   Outside Right:   Right Block (blockType=+1), stimPosition=+1 (right screen)
    isOutsideL  = (blockType == -1) & (stimPos == -1);
    isCenterLB  = (blockType == -1) & (stimPos ==  1);
    isCenterRB  = (blockType ==  1) & (stimPos == -1);
    isOutsideR  = (blockType ==  1) & (stimPos ==  1);

    stimBinsAll = round(stimOn / binSize);
    moveBinsAll = round(moveStart / binSize);

    % --- Initialize predictor matrices ---
    AstimOutsideL = zeros(nT, nK1);
    AstimOutsideR = zeros(nT, nK1);
    AstimCenter   = zeros(nT, nK1);
    AstimLatent   = zeros(nT, nK1);
    Amovement     = zeros(nT, nK2);
    Achoice       = zeros(nT, nK2);
    AlatentMove   = zeros(nT, nK2);

    % --- Stim-aligned predictors ---
    % Each stim kernel bin must occur BEFORE movement onset on that trial.
    % This prevents stimulus representations from bleeding into the movement period.

    % Outside left stimulus (Left Block only)
    trialIdx = find(isOutsideL);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimOutsideL(binPos, kidx) = 1;
            end
        end
    end

    % Outside right stimulus (Right Block only)
    trialIdx = find(isOutsideR);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimOutsideR(binPos, kidx) = 1;
            end
        end
    end

    % Center stimulus: common component (AstimCenter) + latent-modulated (AstimLatent)
    % AstimLatent captures how the neural response to center stimuli varies with the
    % animal's latent context belief. Negative latent → Left Block belief, positive → Right.
    trialIdx = find(isCenterLB);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimCenter(binPos, kidx) = 1;
                AstimLatent(binPos, kidx) = latent(t);
            end
        end
    end

    trialIdx = find(isCenterRB);
    for jj = 1:numel(trialIdx)
        t = trialIdx(jj);
        for kidx = 1:nK1
            binPos = stimBinsAll(t) + kernelLags{1}(kidx);
            if binPos >= 1 && binPos <= nT && binPos <= moveBinsAll(t)
                AstimCenter(binPos, kidx) = 1;
                AstimLatent(binPos, kidx) = latent(t);
            end
        end
    end

    % --- Movement-aligned predictors ---
    % Each movement kernel bin must occur AFTER stimulus onset on that trial.
    % This prevents motor representations from bleeding into the pre-stimulus period.

    for jj = 1:numTrials
        for kidx = 1:nK2
            binPos = moveBinsAll(jj) + kernelLags{2}(kidx);
            if binPos >= 1 && binPos <= nT && binPos >= stimBinsAll(jj)
                Amovement(binPos, kidx)   = 1;
                Achoice(binPos, kidx)     = response(jj);
                AlatentMove(binPos, kidx) = latent(jj);
            end
        end
    end
end
