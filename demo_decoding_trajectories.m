%% demo_decoding_trajectories.m
%
% Demonstration of the decoding and trajectory analyses from:
%   Shaker, Schroeter, Birman, & Steinmetz (2026). The midbrain reticular
%   formation in contextual control of perceptual decisions.
%
% This script walks through two analyses for an example session:
%
%   Part 1: Context decoding accuracy 
%     Decode the latent context variable from pre-stimulus baseline activity
%     using elastic net regression, and compare to pseudosession-based
%     shuffled null distributions.
%
%   Part 2: Coding dimensions and neural trajectories
%     Fit Action, Choice, and Context coding dimensions via cross-validated
%     logistic/linear regression, then project population activity onto
%     these dimensions to visualize trial-type-dependent neural trajectories.
%
% The trajectory plots in the paper (Figures 3A-B, K-L) show median
% statistics computed across all qualifying sessions for a given region.
% This demo shows the general approach applied to a single session.
%
% DEPENDENCIES:
%   - glmnet_matlab: https://github.com/lachioma/glmnet_matlab
%   - npy-matlab: https://github.com/kwikteam/npy-matlab
%   - Statistics and Machine Learning Toolbox
%   - Signal Processing Toolbox
%   - Parallel Computing Toolbox

clearvars; close all;

% --- Add all subdirectories of the analysis code folder to the path ---
codeRoot = 'path/to/analysis/code';
addpath(genpath(codeRoot));

%% ========================================================================
%  1. SETUP
%  ========================================================================

% --- Path to the FigShare data folder ---
dataRoot = 'D:\ShakerEtAl_2026_data';   % <-- EDIT THIS to your local path

% --- Session to analyze ---
mouseName = 'JRS_0022';
dateStr   = '2023-12-13';

% --- Read session info from sessions table ---
sessTable = readtable(fullfile(dataRoot, 'sessionsTable.csv'), 'TextType', 'string');
sessIdx = find(sessTable.mouseName == mouseName & ...
               string(sessTable.dates, 'yyyy-MM-dd') == dateStr);
if isempty(sessIdx)
    sessIdx = find(strcmp(sessTable.mouseName, mouseName));
    if isempty(sessIdx); error('Session not found in sessionsTable.csv'); end
    sessIdx = sessIdx(1);
end
probeNames = {sessTable.probeName_1(sessIdx)};
if ~ismissing(sessTable.probeName_2(sessIdx)) && strlength(sessTable.probeName_2(sessIdx)) > 0
    probeNames{end+1} = sessTable.probeName_2(sessIdx);
end

% --- Regions of interest (substring matches into Allen CCF labels) ---
regionCCF      = {'midbrain reticular nucleus', 'superior colliculus motor', ...
                  'secondary motor', 'caudoputamen'};
regionAcronyms = {'MRF', 'SCm', 'MOs', 'CP'};

% --- Parameters ---
params = struct();
params.minFR_baseline    = 0.05;     % minimum baseline firing rate (Hz)
params.RTlimits          = [0 1];    % reaction time filter, can be looser for analyzing activity in pre-stimulus window 
params.baselinePeriod    = [-0.3 0]; % baseline window relative to stim onset
params.inclRepeats       = true;     % include repeat trials (as in paper)

% Part 1: Context decoding accuracy
params.decoding.alpha       = 0.5;   % elastic net mixing (0=ridge, 1=lasso)
params.decoding.numFolds    = 5;     % outer CV folds
params.decoding.lambdaToUse = 'lambda_min';
params.decoding.numDecodings = 10;   % iterations to average over (paper: 100)
params.decoding.numShuff     = 100;  % pseudosession shuffles (paper: 2000)
params.decoding.decodingInterval = [-0.3 -0.05]; % pre-stim baseline
params.decoding.minNeurons  = 3;     % minimum neurons per region

% Part 2: Coding dimensions and trajectories
params.traj.RTlimits = [0.075 0.4];  % tighter RT window for analyzing activity in post-stimulus epoch
params.traj.alpha           = 0;     % pure ridge regression
params.traj.numFolds        = 25;    % outer CV folds for dimension fitting
params.traj.numFolds_lambda = 5;     % inner CV folds for lambda selection
params.traj.lambdaToUse     = 'lambda_min';
params.traj.standardize     = true;  % z-score columns of TxN before fitting
params.traj.baselineNormalize = true; % divide spike counts by baseline FR
params.traj.includeIntercept = false;
params.traj.binSize         = 0.075; % projection bin width (s)
params.traj.stepSize        = 0.01;  % projection step size (s)
params.traj.minNeurons      = 10;    % minimum neurons per region

% --- Trial type colors ---
% Left Block Outside = dark blue, Left Block Center = dark orange,
% Right Block Center = light blue, Right Block Outside = light orange
trialTypeColors = [0 0.25 0.5; 0.5 0.25 0; 0 0.5 1; 1 0.5 0];
trialTypeLabels = {'LB Outside', 'LB Center', 'RB Center', 'RB Outside'};
trialTypeLineWidths = [2.5 5 5 2.5];

% --- Check for glmnet ---
useGlmnet = exist('glmnet', 'file') == 2;
if useGlmnet
    fprintf('Using glmnet_matlab for regression.\n');
else
    error('glmnet_matlab is required for this demo. Install from: https://github.com/lachioma/glmnet_matlab');
end

%% ========================================================================
%  2. LOAD DATA
%  ========================================================================
fprintf('\nLoading session: %s %s\n', mouseName, dateStr);
[blockTL, sp] = loadSessionData(dataRoot, mouseName, dateStr, probeNames);

%% ========================================================================
%  3. TRIAL FILTERING
%  ========================================================================
% Part 1 (decoding accuracy) and Part 2 (trajectories) use different trial
% inclusion criteria, matching the original analyses.

RT = blockTL.moveTimes(:,1) - blockTL.stimulusOnTimes;

% Part 1: broad inclusion - no repeats, RT filter, correct & incorrect trials
inclTrials_decoding = (blockTL.repeatNumValues == 1) & (blockTL.responseValues ~= 0) & ...
                      (RT >= params.RTlimits(1)) & (RT <= params.RTlimits(2));

% Part 2: no repeats, RT filter, correct only
inclTrials_traj = (blockTL.repeatNumValues == 1) & ...
                  (blockTL.responseValues ~= 0) & ...
                  (RT >= params.traj.RTlimits(1)) & (RT <= params.traj.RTlimits(2));
inclTrialsCorrect = inclTrials_traj & blockTL.hitValues;

fprintf('Decoding trials: %d / %d\n', sum(inclTrials_decoding), numel(inclTrials_decoding));
fprintf('Trajectory trials (correct): %d / %d\n', sum(inclTrialsCorrect), numel(inclTrialsCorrect));

%% ========================================================================
%  4. IDENTIFY NEURONS BY REGION
%  ========================================================================
% For each region, collect qualifying neurons across all probes and shanks.
% Quality filters: spike amplitude noise floor passed (spikeFloorOK) and
% baseline firing rate above threshold.

nProbes = numel(sp);
nShanks = 4;

% Compute baseline firing rates for all neurons (using broadest trial set)
baselineTimes = [blockTL.stimulusOnTimes(inclTrials_decoding) + params.baselinePeriod(1), ...
                 blockTL.stimulusOnTimes(inclTrials_decoding) + params.baselinePeriod(2)];
for pn = 1:nProbes
    for sn = 1:nShanks
        if isempty(sp{pn}(sn).cids); continue; end
        nCids = numel(sp{pn}(sn).cids);
        sp{pn}(sn).baselineFR = zeros(nCids, 1);
        for cc = 1:nCids
            cluSt = sp{pn}(sn).st(sp{pn}(sn).clu == sp{pn}(sn).cids(cc));
            nSpikes = 0;
            totalDur = 0;
            for tt = 1:size(baselineTimes, 1)
                nSpikes = nSpikes + sum(cluSt >= baselineTimes(tt,1) & cluSt < baselineTimes(tt,2));
                totalDur = totalDur + (baselineTimes(tt,2) - baselineTimes(tt,1));
            end
            sp{pn}(sn).baselineFR(cc) = nSpikes / totalDur;
        end
    end
end

% Find neurons in each region
% regionNeurons{rr} is an N x 3 matrix: [cidIndex, probeNum, shankNum]
regionNeurons = cell(numel(regionCCF), 1);
for rr = 1:numel(regionCCF)
    neurons = [];
    for pn = 1:nProbes
        for sn = 1:nShanks
            if isempty(sp{pn}(sn).cids); continue; end
            labels = sp{pn}(sn).cidRegLabels;
            if isnumeric(labels); continue; end  

            for cc = 1:numel(sp{pn}(sn).cids)
                if ischar(labels{cc}) || isstring(labels{cc})
                    matchesRegion = contains(lower(labels{cc}), lower(regionCCF{rr}));
                else
                    matchesRegion = false;
                end
                if matchesRegion && sp{pn}(sn).spikeFloorOK(cc) && ...
                        sp{pn}(sn).baselineFR(cc) >= params.minFR_baseline
                    neurons = [neurons; cc, pn, sn];
                end
            end
        end
    end
    regionNeurons{rr} = neurons;
end

% Report neuron counts and note which regions qualify for each analysis
fprintf('\nNeuron counts by region:\n');
for rr = 1:numel(regionCCF)
    nNeur = size(regionNeurons{rr}, 1);
    decFlag = ''; trajFlag = '';
    if nNeur < params.decoding.minNeurons
        decFlag = sprintf(' (below %d for decoding)', params.decoding.minNeurons);
    end
    if nNeur < params.traj.minNeurons
        trajFlag = sprintf(' (below %d for trajectories)', params.traj.minNeurons);
    end
    fprintf('  %s: %d neurons%s%s\n', regionAcronyms{rr}, nNeur, decFlag, trajFlag);
end

%% ========================================================================
%  PART 1: CONTEXT DECODING ACCURACY
%  ========================================================================
% Decode the latent context variable from pre-stimulus baseline firing
% rates using elastic net regression (alpha = 0.5). The latent variable is derived
% from a fitted behavioral model (see Methods: Behavioral modeling). Cross-
% validated R^2 quantifies decoding accuracy. A null
% distribution is generated by substituting pseudosession-derived latent
% traces that preserve the statistical structure of the latent context variable but
% destroy the precise timing of its transitions in the actual session.

fprintf('\n--- Part 1: Context decoding accuracy ---\n');

decodingInterval = params.decoding.decodingInterval;
numDecodings = params.decoding.numDecodings;
numShuff = params.decoding.numShuff;

% Prepare pseudosession latent values for shuffles
pseudoData = blockTL.pseudosessions_latent;
pseudoSessNums = unique(pseudoData(:, 8));
nShuffToUse = min(numShuff, numel(pseudoSessNums));

% Extract and filter pseudolatent traces to match included trials
numInclTrials_dec = sum(inclTrials_decoding);
pseudolatent_incl = zeros(numInclTrials_dec, nShuffToUse);
for ii = 1:nShuffToUse
    thisSess = pseudoData(pseudoData(:,8) == pseudoSessNums(ii), :);
    nAvail = min(size(thisSess, 1), numInclTrials_dec);
    pseudolatent_incl(1:nAvail, ii) = thisSess(1:nAvail, 6);  % latent values
end
fprintf('Loaded %d pseudosession latent traces for shuffles.\n', nShuffToUse);

% Identify regions with enough neurons for decoding
regionsForDecoding = find(cellfun(@(x) size(x,1), regionNeurons) >= params.decoding.minNeurons);
if isempty(regionsForDecoding)
    fprintf('No regions meet minimum neuron count (%d) for decoding.\n', params.decoding.minNeurons);
end

% Decoding loop
decodingR2    = cell(numel(regionCCF), 1);  % R^2 per region
decodingR     = cell(numel(regionCCF), 1);  % Pearson r per region
decodingR2_shuff = cell(numel(regionCCF), 1);
decodingR_shuff  = cell(numel(regionCCF), 1);

if isempty(gcp('nocreate')) && license('test', 'Distrib_Computing_Toolbox')
    parpool;
end

for rr = regionsForDecoding'
    theseNeurons = regionNeurons{rr};
    nNeurons = size(theseNeurons, 1);
    fprintf('\nDecoding context from %s (%d neurons)...\n', regionAcronyms{rr}, nNeurons);

    % Build TxN matrix: spike counts in pre-stimulus baseline window
    startTimes = blockTL.stimulusOnTimes(inclTrials_decoding) + decodingInterval(1);
    endTimes   = blockTL.stimulusOnTimes(inclTrials_decoding) + decodingInterval(2);
    TxN = produceTxN(theseNeurons, sp, startTimes, endTimes, false, []);
    TxN(:, ~any(TxN, 1)) = [];  % remove silent neurons

    % True labels: rescaled latent context variable
    trueLabels = rescale(blockTL.latent(inclTrials_decoding), -1, 1);

    % --- Real decoding (averaged over numDecodings iterations) ---
    R2_iters = zeros(numDecodings, 1);
    r_iters  = zeros(numDecodings, 1);
    for di = 1:numDecodings
        CVO = cvpartition(numInclTrials_dec, 'KFold', params.decoding.numFolds);
        predLabels = NaN(numInclTrials_dec, 1);
        for fold = 1:params.decoding.numFolds
            trainIdx = CVO.training(fold);
            testIdx  = CVO.test(fold);
            opts = glmnetSet(struct('alpha', params.decoding.alpha, ...
                'standardize', true, 'intr', true, 'maxit', 10000));
            fit = cvglmnet(TxN(trainIdx,:), trueLabels(trainIdx), 'gaussian', ...
                opts, 'deviance', params.decoding.numFolds, [], true);
            predLabels(testIdx) = cvglmnetPredict(fit, TxN(testIdx,:), ...
                params.decoding.lambdaToUse, 'link');
        end
        R2_iters(di) = 1 - sum((trueLabels - predLabels).^2) / sum((trueLabels - mean(trueLabels)).^2); % coefficient of determination
        r_iters(di)  = corr(trueLabels, predLabels); % pearson r
    end
    decodingR2{rr} = mean(R2_iters);
    decodingR{rr}  = mean(r_iters);

    % --- Shuffled decoding (pseudosession labels) ---
    R2_shuff = zeros(nShuffToUse, 1);
    r_shuff  = zeros(nShuffToUse, 1);
    fprintf('  Running %d shuffles...\n', nShuffToUse);
    shuffTimer = tic;
    for si = 1:nShuffToUse
        if mod(si, 25) == 0
            elapsed = toc(shuffTimer);
            fprintf('    Shuffle %d/%d (%.1f min elapsed)\n', si, nShuffToUse, elapsed/60);
        end
        shuffLabels = rescale(pseudolatent_incl(:, si), -1, 1);
        CVO = cvpartition(numInclTrials_dec, 'KFold', params.decoding.numFolds);
        predLabels = NaN(numInclTrials_dec, 1);
        for fold = 1:params.decoding.numFolds
            trainIdx = CVO.training(fold);
            testIdx  = CVO.test(fold);
            opts = glmnetSet(struct('alpha', params.decoding.alpha, ...
                'standardize', true, 'intr', true, 'maxit', 10000));
            fit = cvglmnet(TxN(trainIdx,:), shuffLabels(trainIdx), 'gaussian', ...
                opts, 'deviance', params.decoding.numFolds, [], true);
            predLabels(testIdx) = cvglmnetPredict(fit, TxN(testIdx,:), ...
                params.decoding.lambdaToUse, 'link');
        end
        R2_shuff(si) = 1 - sum((shuffLabels - predLabels).^2) / sum((shuffLabels - mean(shuffLabels)).^2);
        r_shuff(si)  = corr(shuffLabels, predLabels);
    end
    decodingR2_shuff{rr} = R2_shuff;
    decodingR_shuff{rr}  = r_shuff;

    % p-value
    M = sum(decodingR{rr} > r_shuff);
    pVal = (1 + nShuffToUse - M) / (1 + nShuffToUse);
    fprintf('  %s: r = %.3f, R^2 = %.3f, p = %.4f\n', ...
        regionAcronyms{rr}, decodingR{rr}, decodingR2{rr}, pVal);
end

% --- Decoding true vs. pseudosession shuffle null histogram plots ---

if ~isempty(regionsForDecoding)
    figure('WindowState', 'maximized', 'Color', 'w');
    nReg = numel(regionsForDecoding);
    hold on;
    for idx = 1:nReg
        rr = regionsForDecoding(idx);
        
        ax(idx) = subplot(1,nReg,idx);
        shuffDist = decodingR_shuff{rr};
        histogram(shuffDist,-0.5:0.025:1,'FaceColor',[0.5 0.5 0.5],'EdgeColor',[0.5 0.5 0.5])
        xline(decodingR{rr}, '-r', 'LineWidth', 1.5)
        title(regionAcronyms{rr})
        box off
    end
    linkaxes(ax,'xy')
    xlabel(ax(1),'Context decoding score (Pearson r)');
    ylabel(ax(1),'Number of shuffles')
    sgtitle(sprintf('Context decoding: %s %s', mouseName, dateStr),'Interpreter','None');
    theseYlim = ylim;
    text(ax(1), decodingR{regionsForDecoding(1)} + 0.05, theseYlim(2)-3, 'True score', 'HorizontalAlignment', 'left', 'FontSize', 15, 'Color', 'r');

    
    for idx = 1:nReg
        rr = regionsForDecoding(idx);
       % Significance marker
        M = sum(decodingR{rr} > decodingR_shuff{rr});
        pVal = (1 + numel(decodingR_shuff{rr}) - M) / (1 + numel(decodingR_shuff{rr}));
        if pVal < 0.001
            text(ax(idx), decodingR{rr} + 0.1, theseYlim(2)-1, '***', 'HorizontalAlignment', 'left', 'FontSize', 30);
        elseif pVal < 0.01
            text(ax(idx), decodingR{rr} + 0.1, theseYlim(2)-1, '**', 'HorizontalAlignment', 'left', 'FontSize', 30);
        elseif pVal < 0.05
            text(ax(idx), decodingR{rr} + 0.1, theseYlim(2)-1, '*', 'HorizontalAlignment', 'left', 'FontSize', 30);
        end 
    end
end
clear ax

%% ========================================================================
%  PART 2: CODING DIMENSIONS AND NEURAL TRAJECTORIES
%  ========================================================================
% Fit three coding dimensions via cross-validated regression:
%   - CW action dimension: classifies baseline vs. pre-movement activity
%     on CW-choice trials (L2-regularized logistic regression)
%   - CCW action dimension: same, for CCW-choice trials
%   - Context dimension: classifies pre-stimulus activity by block identity
%     (L2-regularized logistic regression)
%
% The Action dimension is defined as CW + CCW (choice-invariant movement
% signal) and the Choice dimension as CW - CCW. Population activity is
% projected onto these three dimensions using held-out test trials from the
% same CV partition, yielding neural trajectories in Action x Choice x
% Context space.

fprintf('\n--- Part 2: Coding dimensions and trajectories ---\n');

% Identify regions with enough neurons for trajectories
regionsForTraj = find(cellfun(@(x) size(x,1), regionNeurons) >= params.traj.minNeurons);
if isempty(regionsForTraj)
    fprintf('No regions meet minimum neuron count (%d) for trajectories.\n', params.traj.minNeurons);
end

% Trajectory analyses use correct trials only
RT_correct = blockTL.moveTimes(inclTrialsCorrect, 1) - blockTL.stimulusOnTimes(inclTrialsCorrect);
numCorrectTrials = sum(inclTrialsCorrect);

% Shared CV partition across all dimension fits
cvPartition = cvpartition(numCorrectTrials, 'KFold', params.traj.numFolds);

% Trial type assignment for trajectory visualization
btCorr = blockTL.blockType(inclTrialsCorrect);
spCorr = blockTL.stimPosition(inclTrialsCorrect);
trialTypes = zeros(numCorrectTrials, 1);
trialTypes(btCorr == -1 & spCorr == -1) = 1;  % Left Block, Outside Left
trialTypes(btCorr == -1 & spCorr ==  1) = 2;  % Left Block, Center
trialTypes(btCorr ==  1 & spCorr == -1) = 3;  % Right Block, Center
trialTypes(btCorr ==  1 & spCorr ==  1) = 4;  % Right Block, Outside Right

for rr = regionsForTraj'
    theseNeurons = regionNeurons{rr};
    nNeurons = size(theseNeurons, 1);
    fprintf('\nFitting coding dimensions for %s (%d neurons)...\n', regionAcronyms{rr}, nNeurons);

    % ====================================================================
    %  2a. Fit Context dimension (pre-stimulus baseline → block identity)
    % ====================================================================
    startTimes_ctx = blockTL.stimulusOnTimes(inclTrialsCorrect) + params.decoding.decodingInterval(1);
    endTimes_ctx   = blockTL.stimulusOnTimes(inclTrialsCorrect) + params.decoding.decodingInterval(2);
    TxN_ctx = produceTxN(theseNeurons, sp, startTimes_ctx, endTimes_ctx, ...
        params.traj.baselineNormalize, sp);
    if params.traj.standardize
        activeCols = any(TxN_ctx, 1);
        TxN_ctx(:, activeCols) = normalize(TxN_ctx(:, activeCols), 1, 'zscore');
    end

    trueLabels_ctx = zeros(numCorrectTrials, 1);
    trueLabels_ctx(btCorr == -1) = 1;
    trueLabels_ctx(btCorr ==  1) = 2;

    contextWeights = zeros(nNeurons + 1, params.traj.numFolds);
    for fold = 1:params.traj.numFolds
        trainIdx = find(cvPartition.training(fold));

        numUniqueLabels = 2;
        observationWeights = ones(numel(trainIdx), 1);
        for cLabel = [1 2]
            nLabel = sum(trueLabels_ctx(trainIdx) == cLabel);
            observationWeights(trueLabels_ctx(trainIdx) == cLabel) = ...
                numel(trainIdx) / (numUniqueLabels * nLabel);
        end

        opts = glmnetSet(struct('alpha', params.traj.alpha, ...
            'standardize', false, 'intr', params.traj.includeIntercept, ...
            'maxit', 10000, 'weights', observationWeights));
        fit = cvglmnet(TxN_ctx(trainIdx,:), trueLabels_ctx(trainIdx), 'binomial', ...
            opts, 'deviance', params.traj.numFolds_lambda, [], true);
        contextWeights(:, fold) = cvglmnetCoef(fit, params.traj.lambdaToUse);
    end
    fprintf('  Context dimension fitted.\n');

    % ====================================================================
    %  2b. Fit CW and CCW action dimensions
    % ====================================================================
    % Each classifies baseline (-300 to 50 ms pre-stim) vs. pre-movement
    % (stim onset to movement onset) activity, restricted to trials of the
    % corresponding choice direction. The TxN matrix is stacked:
    % [baseline trials; pre-movement trials] with labels [1; 2].

    startTimes_base = blockTL.stimulusOnTimes(inclTrialsCorrect) - 0.3;
    endTimes_base   = blockTL.stimulusOnTimes(inclTrialsCorrect) - 0.05;
    startTimes_pre  = blockTL.stimulusOnTimes(inclTrialsCorrect);
    endTimes_pre    = blockTL.moveTimes(inclTrialsCorrect, 1);

    TxN_base = produceTxN(theseNeurons, sp, startTimes_base, endTimes_base, ...
        params.traj.baselineNormalize, sp);
    TxN_pre  = produceTxN(theseNeurons, sp, startTimes_pre, endTimes_pre, ...
        params.traj.baselineNormalize, sp);
    TxN_stacked = [TxN_base; TxN_pre];
    if params.traj.standardize
        activeCols = any(TxN_stacked, 1);
        TxN_stacked(:, activeCols) = normalize(TxN_stacked(:, activeCols), 1, 'zscore');
    end
    stackedLabels = [ones(numCorrectTrials, 1); repelem(2, numCorrectTrials, 1)];

    respCorr = blockTL.responseValues(inclTrialsCorrect);
    cwTrials  = respCorr == 1;   % CW choice
    ccwTrials = respCorr == -1;  % CCW choice

    cwWeights  = zeros(nNeurons + 1, params.traj.numFolds);
    ccwWeights = zeros(nNeurons + 1, params.traj.numFolds);

    for fold = 1:params.traj.numFolds
        thisTrain = cvPartition.training(fold);

        for choiceDir = 1:2
            if choiceDir == 1
                choiceFilter = cwTrials;
            else
                choiceFilter = ccwTrials;
            end
            trainIdx = find(thisTrain & choiceFilter);
            trainIdx = [trainIdx; trainIdx + numCorrectTrials];

            numUniqueLabels = 2;
            observationWeights = ones(numel(trainIdx), 1);
            for cLabel = [1 2]
                nLabel = sum(stackedLabels(trainIdx) == cLabel);
                observationWeights(stackedLabels(trainIdx) == cLabel) = ...
                    numel(trainIdx) / (numUniqueLabels * nLabel);
            end

            opts = glmnetSet(struct('alpha', params.traj.alpha, ...
                'standardize', false, 'intr', params.traj.includeIntercept, ...
                'maxit', 10000, 'weights', observationWeights));
            fit = cvglmnet(TxN_stacked(trainIdx,:), stackedLabels(trainIdx), ...
                'binomial', opts, 'deviance', params.traj.numFolds_lambda, [], true);
            w = cvglmnetCoef(fit, params.traj.lambdaToUse);

            if choiceDir == 1
                cwWeights(:, fold) = w;
            else
                ccwWeights(:, fold) = w;
            end
        end
    end
    fprintf('  CW and CCW action dimensions fitted.\n');

    % ====================================================================
    %  2c. Project onto Action x Choice x Context dimensions
    % ====================================================================
    % Action = CW + CCW (choice-invariant movement signal)
    % Choice = CW - CCW (choice-selective signal)
    % Context = block classification weights

    binSize  = params.traj.binSize;
    stepSize = params.traj.stepSize;

    % Stim-aligned projection windows
    stimAligned = -binSize : stepSize : 0.25 - binSize;
    stimIntervals = [stimAligned; stimAligned + binSize];

    % Move-aligned projection windows
    moveAligned = -0.25 - binSize : stepSize : -binSize;
    moveIntervals = [moveAligned; moveAligned + binSize];

    nStimBins = size(stimIntervals, 2);
    nMoveBins = size(moveIntervals, 2);
    nTotalBins = nStimBins + nMoveBins;

    neurDims = 2:size(contextWeights, 1);  % excludes intercept row

    testSetProj = NaN(numCorrectTrials, nTotalBins, 3);

    for fold = 1:params.traj.numFolds
        actionChoiceContextDims = [ ...
            cwWeights(neurDims, fold) + ccwWeights(neurDims, fold), ...   % Action
            cwWeights(neurDims, fold) - ccwWeights(neurDims, fold), ...   % Choice
            contextWeights(neurDims, fold)];                              % Context

        testTrialIdx = find(cvPartition.test(fold));
        trainTrialIdx = find(cvPartition.training(fold));

        trainSetProj = cell(nTotalBins, 1);
        for tBin = 1:nTotalBins
            if tBin <= nStimBins
                theseTimes = blockTL.stimulusOnTimes(inclTrialsCorrect);
                theseIntervals = stimIntervals;
                binIdx = tBin;
            else
                theseTimes = blockTL.moveTimes(inclTrialsCorrect, 1);
                theseIntervals = moveIntervals;
                binIdx = tBin - nStimBins;
            end
            startT = theseTimes + theseIntervals(1, binIdx);
            endT   = theseTimes + theseIntervals(2, binIdx);

            TxN_bin = produceTxN(theseNeurons, sp, startT, endT, ...
                params.traj.baselineNormalize, sp);

            testSetProj(testTrialIdx, tBin, :) = ...
                TxN_bin(testTrialIdx, :) * actionChoiceContextDims;
            trainSetProj{tBin} = TxN_bin(trainTrialIdx, :) * actionChoiceContextDims;
        end

        % Sign correction: flip Action axis if baseline values exceed
        % end values (ensures Action increases from baseline to movement)
        if mean(trainSetProj{1}(:, 1), 'omitnan') > mean(trainSetProj{end}(:, 1), 'omitnan')
            testSetProj(testTrialIdx, :, 1) = testSetProj(testTrialIdx, :, 1) * -1;
        end
    end

    fprintf('  Projected onto %d time bins (%d stim-aligned, %d move-aligned).\n', ...
        nTotalBins, nStimBins, nMoveBins);

    % ====================================================================
    %  2d. Visualize trajectories
    % ====================================================================
    % Trajectories visualized per region for an example session. Note that 
    % single-session trajectories will appear noisier than the trajectories
    % in the paper, which are combined and then averaged across sessions. 

    
    % Robust z-score normalization per dimension (across all trials and
    % time bins jointly) for visualization. In the paper, this is done per
    % session before combining sessions so as to bring them into comparable
    % units. 
    projNorm = testSetProj;
    for dim = 1:3
        projNorm(:,:,dim) = robust_z_scale(testSetProj(:,:,dim));
    end

    % Plot move-aligned bins only
    binsToPlot = (nStimBins + 1) : nTotalBins;
    nPlotBins = numel(binsToPlot);

    % Compute trial-type-averaged trajectories over plotted bins
    trajMean = zeros(nPlotBins, 3, 4);  % timeBins x dims x trialTypes
    trajSEM  = zeros(nPlotBins, 3, 4);
    for ti = 1:4
        theseTrials = trialTypes == ti;
        for dim = 1:3
            trajMean(:, dim, ti) = mean(projNorm(theseTrials, binsToPlot, dim), 1, 'omitnan')';
            trajSEM(:, dim, ti)  = std(projNorm(theseTrials, binsToPlot, dim), [], 1, 'omitnan')' / sqrt(sum(theseTrials));
        end
    end

    % Line widths increase over time to indicate temporal progression
    lineWidths = linspace(1, 5, nPlotBins - 1);

    % --- Figure setup: 2x2 grid ---
    axisLabels = {'Action', 'Choice', 'Context'};
    figure('Position', [50 50 1200 900], 'Color', 'w');
    sgtitle(sprintf('%s  %s  %s', ...
        mouseName, dateStr, regionAcronyms{rr}), ...
        'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'None');

    ax = gobjects(4, 1);
    ax(1) = subplot(2, 2, 1);  % 3D trajectory
    ax(2) = subplot(2, 2, 2);  % Action vs Choice
    ax(3) = subplot(2, 2, 3);  % Choice vs Context
    ax(4) = subplot(2, 2, 4);  % Action vs Context
    hold(ax(1), 'on'); hold(ax(2), 'on'); hold(ax(3), 'on'); hold(ax(4), 'on');

    % Dimension pairings for 2D projections: [xDim yDim] per subplot
    projPairs = {[1 2], [2 3], [1 3]};  % ax(2): A vs Ch, ax(3): Ch vs Ctx, ax(4): A vs Ctx

    for ti = 1:4
        pts = squeeze(trajMean(:, :, ti));   % nPlotBins x 3
        err = squeeze(trajSEM(:, :, ti));

        % Start markers
        scatter3(ax(1), pts(1,1), pts(1,2), pts(1,3), 100, trialTypeColors(ti,:), 'x', 'LineWidth', 2);
        for pp = 1:3
            d1 = projPairs{pp}(1); d2 = projPairs{pp}(2);
            scatter(ax(pp+1), pts(1,d1), pts(1,d2), 100, trialTypeColors(ti,:), 'x', 'LineWidth', 2);
        end

        % Trajectory segments
        for jj = 1:nPlotBins - 1
            % 3D trajectory
            plot3(ax(1), pts(jj:jj+1, 1), pts(jj:jj+1, 2), pts(jj:jj+1, 3), ...
                '-', 'Color', trialTypeColors(ti,:), 'LineWidth', lineWidths(jj));

            % 2D projections with error bars
            for pp = 1:3
                d1 = projPairs{pp}(1); d2 = projPairs{pp}(2);
                plot(ax(pp+1), pts(jj:jj+1, d1), pts(jj:jj+1, d2), ...
                    '-', 'Color', trialTypeColors(ti,:), 'LineWidth', lineWidths(jj));
                errorbar(ax(pp+1), pts(jj,d1), pts(jj,d2), ...
                    err(jj,d2), err(jj,d2), err(jj,d1), err(jj,d1), ...
                    'o', 'Color', trialTypeColors(ti,:), 'LineWidth', 1, ...
                    'CapSize', 0, 'MarkerSize', 3, 'MarkerFaceColor', trialTypeColors(ti,:));
            end
        end

        % Error bar on final point
        for pp = 1:3
            d1 = projPairs{pp}(1); d2 = projPairs{pp}(2);
            errorbar(ax(pp+1), pts(end,d1), pts(end,d2), ...
                err(end,d2), err(end,d2), err(end,d1), err(end,d1), ...
                'o', 'Color', trialTypeColors(ti,:), 'LineWidth', 1, ...
                'CapSize', 0, 'MarkerSize', 3, 'MarkerFaceColor', trialTypeColors(ti,:));
        end

        % End markers
        scatter3(ax(1), pts(end,1), pts(end,2), pts(end,3), 60, trialTypeColors(ti,:), 'o', 'filled');
        for pp = 1:3
            d1 = projPairs{pp}(1); d2 = projPairs{pp}(2);
            scatter(ax(pp+1), pts(end,d1), pts(end,d2), 60, trialTypeColors(ti,:), 'o', 'filled');
        end
    end

    % Axis formatting
    view(ax(1), -35.264, 45); % isometric
    grid(ax(1), 'on');
    xlabel(ax(1), axisLabels{1}); ylabel(ax(1), axisLabels{2}); zlabel(ax(1), axisLabels{3});
    for pp = 1:3
        d1 = projPairs{pp}(1); d2 = projPairs{pp}(2);
        xlabel(ax(pp+1), axisLabels{d1}); ylabel(ax(pp+1), axisLabels{d2});
        box(ax(pp+1), 'off'); set(ax(pp+1), 'TickDir', 'out');
    end
end

fprintf('\nDone.\n');