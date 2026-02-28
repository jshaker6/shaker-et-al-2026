%% demo_GLM_encoding.m
%
% Demonstration of the kernel regression encoding model from:
%   Shaker, Schroeter, Birman, & Steinmetz (2026). The midbrain reticular
%   formation in contextual control of perceptual decisions.
%
% This script walks through the encoding model pipeline for a single session:
%   1) Load data from the FigShare repository (ShakerEtAl_2026)
%   2) Preprocess spikes (bin, baseline-normalize, causal smooth)
%   3) Construct task event predictor matrices with time-lagged kernels
%   4) Reduce predictor dimensionality via reduced rank regression (RRR)
%   5) Fit ridge regression to predict each neuron's activity
%   6) Assess variable significance via pseudosession-based shuffling
%
% The encoding model fits time-lagged kernels for task variables to each
% neuron's firing rate. To determine which variables significantly modulate
% a neuron's activity, each variable group is independently replaced with a
% shuffled version and the full model is refit. The change in cross-validated
% R^2 (deltaR2 = R2_real - median(R2_shuffled)) quantifies each variable's
% unique encoding strength. The shuffled model has the same
% number of predictors as the real model, but the target variable's
% precise timing is destroyed, while preserving its broader statistical properties. 
% p-values are computed nonparametrically from the empirical null distribution 
% of shuffled R^2 values.
%
% DEPENDENCIES:
%   - glmnet_matlab (recommended): https://github.com/lachioma/glmnet_matlab
%     Provides fast elastic net / ridge regression with cross-validated lambda.
%     If unavailable, the script falls back to MATLAB's built-in lasso().
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
mouseName = 'JRS_0028';
dateStr   = '2023-08-07';

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

% --- Which probe to analyze (GLM is fit per probe, across all 4 shanks) ---
probeToAnalyze = 1;  % index into probeNames

% --- Check for glmnet ---
useGlmnet = exist('glmnet', 'file') == 2;
if useGlmnet
    fprintf('Using glmnet_matlab for ridge regression.\n');
else
    fprintf('glmnet_matlab not found -- using MATLAB lasso() fallback (slower).\n');
    fprintf('For best performance, install: https://github.com/lachioma/glmnet_matlab\n');
end

% --- Parameters ---
params = struct();
params.binSize       = 0.01;       % 10 ms time bins
params.stdevSmoothing = 0.025;     % causal half-Gaussian smoothing (25 ms SD)
params.baselinePeriod = [-0.3 0];  % baseline: 300 ms before stimulus onset
params.RTlimits      = [0.075 0.4];% reaction time inclusion window (seconds)
params.minFR         = 0.1;        % minimum firing rate threshold (Hz)
params.trialMod_pThresh = 0.15;    % trial modulation p-value (Bonferroni-corrected)
params.numShuff      = 50;         % pseudosession shuffles (paper used 500, 
                                   % which takes 1-2 hours to run per session
                                   % but provides much more reliable significance testing)
params.nPCs          = 16;         % face motion SVD components (Stringer et al. 2019)
params.numRanks      = 20;         % reduced rank regression dimensions
params.numFolds      = 5;          % K-fold cross-validation
params.lagTimes      = [0 0.4;     % stim kernel: 0 to max RT
                        -0.25 0];  % move kernel: -250 ms to movement onset

kernelLags{1} = floor(params.lagTimes(1,1)/params.binSize):ceil(params.lagTimes(1,2)/params.binSize);
kernelLags{2} = floor(params.lagTimes(2,1)/params.binSize):ceil(params.lagTimes(2,2)/params.binSize);

%% ========================================================================
%  2. LOAD DATA
%  ========================================================================
fprintf('\nLoading session: %s %s\n', mouseName, dateStr);
[blockTL, sp, motSVD, pupilCoords] = loadSessionData(dataRoot, mouseName, dateStr, probeNames);

expStartStop = blockTL.expStartStop;

%% ========================================================================
%  3. TRIAL FILTERING
%  ========================================================================
% Include non-repeat trials with responses and reaction times within limits.

RT = blockTL.moveTimes(:,1) - blockTL.stimulusOnTimes;
inclTrials = (blockTL.repeatNumValues == 1) & ...
             (blockTL.responseValues ~= 0) & ...
             (RT >= params.RTlimits(1)) & (RT <= params.RTlimits(2));

numTrials = sum(inclTrials);
fprintf('Included %d / %d trials after filtering.\n', numTrials, numel(inclTrials));

%% ========================================================================
%  4. NEURON SELECTION
%  ========================================================================
% Two-stage selection:
%   a) Firing rate threshold: mean trial FR >= 0.1 Hz, quality control
%      metrics passed (refractory period and spike amplitude noise floor)
%   b) Trial modulation: at least one of 6 statistical tests must be
%      significant using a loose threshold after Bonferroni correction (p < 0.15/6). 
%      Tests assess stimulus-evoked, movement-related, choice-selective,
%      block-selective, and stimulus-position-selective modulation.

pn = probeToAnalyze;
nShanks = numel(sp{pn});

% Trial time windows for FR computation and modulation tests
trialTimes = [blockTL.stimulusOnTimes(inclTrials) blockTL.moveTimes(inclTrials,1)];
baselineTimes = [blockTL.stimulusOnTimes(inclTrials) + params.baselinePeriod(1), ...
                 blockTL.stimulusOnTimes(inclTrials) + params.baselinePeriod(2)];
stimWindow = 0.125;
stimTimes = [blockTL.stimulusOnTimes(inclTrials), blockTL.stimulusOnTimes(inclTrials) + stimWindow];
moveWindow = 0.1;
preMoveTimes = [blockTL.moveTimes(inclTrials,1) - moveWindow, blockTL.moveTimes(inclTrials,1)];

% Choice-split pre-movement windows
resp = blockTL.responseValues;
choicePreMoveTimes{1} = [blockTL.moveTimes(inclTrials & resp==-1, 1) - moveWindow, ...
                         blockTL.moveTimes(inclTrials & resp==-1, 1)];
choicePreMoveTimes{2} = [blockTL.moveTimes(inclTrials & resp==1, 1) - moveWindow, ...
                         blockTL.moveTimes(inclTrials & resp==1, 1)];

% Block-split trial windows
bt = blockTL.blockType;
blockTrialTimes{1} = [blockTL.stimulusOnTimes(inclTrials & bt==-1), blockTL.moveTimes(inclTrials & bt==-1, 1)];
blockTrialTimes{2} = [blockTL.stimulusOnTimes(inclTrials & bt==1),  blockTL.moveTimes(inclTrials & bt==1, 1)];

% Stimulus position windows (3 screen locations)
sp_field = blockTL.stimPosition; bt_field = blockTL.blockType;
isOutL = (bt_field==-1) & (sp_field==-1);
isCent = ((bt_field==-1) & (sp_field==1)) | ((bt_field==1) & (sp_field==-1));
isOutR = (bt_field==1) & (sp_field==1);
stimPosTimes{1} = [blockTL.stimulusOnTimes(inclTrials & isOutL), blockTL.stimulusOnTimes(inclTrials & isOutL) + stimWindow];
stimPosTimes{2} = [blockTL.stimulusOnTimes(inclTrials & isCent), blockTL.stimulusOnTimes(inclTrials & isCent) + stimWindow];
stimPosTimes{3} = [blockTL.stimulusOnTimes(inclTrials & isOutR), blockTL.stimulusOnTimes(inclTrials & isOutR) + stimWindow];

cidInclIdxs = cell(nShanks, 1);
for sn = 1:nShanks
    if isempty(sp{pn}(sn).cids); cidInclIdxs{sn} = []; continue; end

    % Compute firing rates
    [sp{pn}(sn).fr, ~] = calcAvgTrialFR(sp{pn}(sn), trialTimes);
    [sp{pn}(sn).baselineFR, ~] = calcAvgTrialFR(sp{pn}(sn), baselineTimes);
    FRpass = sp{pn}(sn).fr >= params.minFR;
    cidGood = FRpass & sp{pn}(sn).refracOK & sp{pn}(sn).spikeFloorOK;
    sp{pn}(sn).FRpass = sp{pn}(sn).cids(FRpass);
    sp{pn}(sn).cidGood = sp{pn}(sn).cids(cidGood);
    allPassIdx = find(cidGood);

    if isempty(allPassIdx); cidInclIdxs{sn} = []; continue; end

    % Trial modulation tests
    [~, bFR]  = calcAvgTrialFR(sp{pn}(sn), baselineTimes);
    [~, tFR]  = calcAvgTrialFR(sp{pn}(sn), trialTimes);
    [~, sFR]  = calcAvgTrialFR(sp{pn}(sn), stimTimes);
    [~, pmFR] = calcAvgTrialFR(sp{pn}(sn), preMoveTimes);
    for jj = 1:2; [~, cpmFR{jj}] = calcAvgTrialFR(sp{pn}(sn), choicePreMoveTimes{jj}); end
    for jj = 1:2; [~, btFR{jj}]  = calcAvgTrialFR(sp{pn}(sn), blockTrialTimes{jj}); end
    for jj = 1:3; [~, spFR{jj}]  = calcAvgTrialFR(sp{pn}(sn), stimPosTimes{jj}); end

    allStimPosFR = [spFR{1} spFR{2} spFR{3}];
    allStimPosGroups = [ones(1,size(spFR{1},2)), 2*ones(1,size(spFR{2},2)), 3*ones(1,size(spFR{3},2))];

    numTests = 6;
    correctedP = params.trialMod_pThresh / numTests;
    keepIdx = false(numel(allPassIdx), 1);
    for ii = 1:numel(allPassIdx)
        cidx = allPassIdx(ii);
        p = zeros(1, numTests);
        p(1) = signrank(bFR(cidx,:), tFR(cidx,:));
        p(2) = signrank(bFR(cidx,:), sFR(cidx,:));
        p(3) = signrank(bFR(cidx,:), pmFR(cidx,:));
        p(4) = ranksum(cpmFR{1}(cidx,:), cpmFR{2}(cidx,:));
        p(5) = ranksum(btFR{1}(cidx,:), btFR{2}(cidx,:));
        p(6) = kruskalwallis(allStimPosFR(cidx,:), allStimPosGroups, 'off');
        keepIdx(ii) = any(p < correctedP);
    end
    cidInclIdxs{sn} = allPassIdx(keepIdx);
end

totalNeurons = sum(cellfun(@numel, cidInclIdxs));
totalCidGood = numel(cat(1, sp{pn}.cidGood));
fprintf('Kept %d / %d neurons after trial modulation filter.\n', totalNeurons, totalCidGood);

%% ========================================================================
%  5. SPIKE PREPROCESSING
%  ========================================================================
%  Bin and smooth spikes with a causal half-gaussian filter

[~, binnedSpNormSmoothed] = baselineNormAndSmooth(sp{pn}, params.binSize, expStartStop, params.stdevSmoothing);

%% ========================================================================
%  6. CONSTRUCT PREDICTOR MATRICES
%  ========================================================================
% Eight task variable predictors:
%   1. AstimOutsideL  - Outside left stimulus (stim-aligned kernel)
%   2. AstimOutsideR  - Outside right stimulus (stim-aligned kernel)
%   3. AstimCenter    - Center stimulus(stim-aligned)
%   4. Amovement      - Action, unsigned (movement-aligned kernel)
%   5. Achoice        - Choice, signed CW/CCW (movement-aligned)
%   6. AlatentMove    - Premotor response modulated by latent context value (movement-aligned)
%   7. AstimLatent    - Center stimulus response modulated by latent context value (stim-aligned)
%   8. Achoice .* AlatentMove - Choice x context interaction (movement-aligned)
%
% Plus nuisance regressors: linear time ramp, tonic baseline effects by block,
% pupil position (2D), and face motion SVD components (16 PCs).

[AstimOutsideL, AstimOutsideR, AstimCenter, AstimLatent, Amovement, Achoice, AlatentMove] = ...
    genPredictorsRevCTask(blockTL, inclTrials, params.binSize, expStartStop, kernelLags);

% Analog predictors (face motion and pupil position)
hasAnalog = ~isempty(motSVD) && ~isempty(pupilCoords);
if hasAnalog
    [ApupilPosition, Aface] = genAnalogPredictors(blockTL.tDigitalUp, pupilCoords, motSVD, params.nPCs, params.binSize, expStartStop);
end

% Assemble model cell array
models = cell(8, 1);
models{1} = AstimOutsideL;
models{2} = AstimOutsideR;
models{3} = AstimCenter;
models{4} = Amovement;
models{5} = Achoice;
models{6} = AlatentMove;
models{7} = AstimLatent;
models{8} = Achoice .* AlatentMove;  % context-choice interaction

% Remove empty kernel columns (stim kernels truncated at movement onset)
for ii = 1:numel(models)
    models{ii} = models{ii}(:, any(models{ii}, 1));
end

% Model groupings: which predictors belong to each tested variable
modelNames    = {'Stim', 'Choice', 'Stim Context', 'Move Context', 'Context (combined)'};
modelGroupings = {[1,2,3], [5,8], 7, [6,8], [6,7,8]};
analyzeBins    = [1 2 1 2 3];  % 1=stim-aligned, 2=move-aligned, 3=both

fprintf('Predictor matrix constructed: %d task variable models.\n', numel(models));

%% ========================================================================
%  7. CROSS-VALIDATION SPLIT AND PREDICTOR ASSEMBLY
%  ========================================================================

[trainSet, testSet, allPredBins, nT, trialNums] = createCVsplit_KFold(...
    expStartStop, params.binSize, ...
    blockTL.stimulusOnTimes(inclTrials), blockTL.moveTimes(inclTrials,1), ...
    kernelLags, params.numFolds);

% Nuisance regressors
Atime = (1:nT)';
AcontextBaseline = zeros(nT, 1);
latent = rescale(blockTL.latent(inclTrials), -1, 1);
for ii = 1:numTrials
    AcontextBaseline(trialNums == ii) = latent(ii);
end

% Subset and clean analog predictors
if hasAnalog
    ApupilPosition = ApupilPosition(allPredBins, :);
    Aface = Aface(allPredBins, :);
    % Robust z-score and outlier removal to handle video artifacts
    ApupilPosition = normalize(ApupilPosition, 1, 'zscore', 'robust');
    ApupilPosition = filloutliers(ApupilPosition, 'linear', 'median', 1);
    for ii = 1:size(ApupilPosition,2); ApupilPosition(:,ii) = rescale(ApupilPosition(:,ii), -1, 1); end
    Aface = filloutliers(Aface, 'linear', 'median', 1);
    for ii = 1:size(Aface,2); Aface(:,ii) = rescale(Aface(:,ii), -1, 1); end
else
    ApupilPosition = []; Aface = [];
end

% Full predictor matrix: [time, context_baseline, task_models, pupil, face]
% Track which columns of the combined matrix belong to each individual model,
% so we can later extract per-variable kernel weights from the RRR coefficients.
combinedTaskModels = [models{:}];
modelIndices = cell(numel(models), 1);
colOffset = 0;
for ii = 1:numel(models)
    nCols = size(models{ii}, 2);
    modelIndices{ii} = colOffset + (1:nCols);
    colOffset = colOffset + nCols;
end
individualModelNames = {'Stim Outside L', 'Stim Outside R', 'Stim Center', ...
                        'Movement', 'Choice', 'Context (move)', ...
                        'Context (stim)', 'Choice x Context'};

Pfull = [Atime, AcontextBaseline, combinedTaskModels(allPredBins,:), ApupilPosition, Aface];
nNuisanceCols = 2;  % Atime + AcontextBaseline (precede task models in Pfull)

% Concatenate neural data across shanks
concatNeurons = {};
for sn = 1:nShanks
    if ~isempty(cidInclIdxs{sn})
        concatNeurons{end+1} = binnedSpNormSmoothed{sn}(:, cidInclIdxs{sn});
    end
end
allNeurons = horzcat(concatNeurons{:});
allNeurons = allNeurons(allPredBins, :);
nNeurons = size(allNeurons, 2);

fprintf('%d neurons, %d time bins, %d predictors.\n', nNeurons, nT, size(Pfull, 2));

%% ========================================================================
%  8. REDUCED RANK REGRESSION
%  ========================================================================
% Project the high-dimensional predictor matrix into a lower-dimensional
% space that captures the most neural variance . This reduces overfitting
% from the large number of predictors.

[~, bComb, ~, ~] = CanonCor2(allNeurons, Pfull);
reducedRanks = Pfull * bComb(:, 1:params.numRanks);

fprintf('RRR complete: %d predictors → %d reduced-rank dimensions.\n', size(Pfull,2), params.numRanks);

%% ========================================================================
%  9. FIT FULL MODEL (ALL NEURONS)
%  ========================================================================
% Ridge regression on the reduced-rank predictors, with lambda selected via
% nested cross-validation,

if useGlmnet && license('test', 'Distrib_Computing_Toolbox') && isempty(gcp('nocreate')); parpool; end % start parpool for Glmnet

numModels = numel(modelNames);
varEtest_real = zeros(nNeurons, numModels);
varEtest_full = zeros(nNeurons, 1);
lambdaAll = zeros(nNeurons, params.numFolds);
xAll = cell(nNeurons, 1);        % regression coefficients (for kernel extraction)
predBAll = zeros(nT, nNeurons);   % full model predictions (for diagnostic PSTHs)

fprintf('\nFitting full model for %d neurons...\n', nNeurons);
neuronIdx = 0;  % global neuron counter
for sn = 1:nShanks
    for jj = 1:numel(cidInclIdxs{sn})
        neuronIdx = neuronIdx + 1;
        B = binnedSpNormSmoothed{sn}(allPredBins, cidInclIdxs{sn}(jj));

        % Fit full model with CV lambda selection
        [thisX, varEtest_full(neuronIdx), thislambda, predB] = ...
            fitRidgeEncoding(reducedRanks, B, trainSet, testSet, true(nT,1), useGlmnet);
        lambdaAll(neuronIdx, :) = thislambda;
        xAll{neuronIdx} = thisX;
        predBAll(:, neuronIdx) = predB;

        % Compute sub-model R^2 on relevant bins only.
        % Each sub-model's R^2 is evaluated using the FULL model's predictions,
        % but restricted to time bins where that sub-model's predictors are active.
        for m = 1:numModels
            modelSubset = [models{modelGroupings{m}}];
            theseBins = any(modelSubset(allPredBins,:), 2);
            varEtest_real(neuronIdx, m) = 1 - ...
                (sum((B(theseBins) - predB(theseBins)).^2) / ...
                 sum((B(theseBins) - mean(B(theseBins))).^2));
        end
    end
end

fprintf('Full model median R^2 = %.3f\n', median(varEtest_full, 'omitnan'));

%% ========================================================================
%  10. EXTRACT PSEUDOLATENT VALUES FOR SHUFFLES
%  ========================================================================
% Pseudosessions are pre-generated simulated behavioral sessions run through
% the fitted context belief model. Each provides a plausible latent context
% trace with the same statistical properties as the real one, but without
% the precise temporal structure.

pseudoData = blockTL.pseudosessions_latent;
pseudoSessNums = unique(pseudoData(:, 8));
nPseudoAvailable = numel(pseudoSessNums);
nShuffToUse = min(params.numShuff, nPseudoAvailable);

pseudolatent_incl = zeros(numTrials, nShuffToUse);
for ii = 1:nShuffToUse
    thisSess = pseudoData(pseudoData(:,8) == pseudoSessNums(ii), :);
    thisSess = thisSess(thisSess(:,4) == 1, :);  % non-repeat trials only
    nAvail = min(size(thisSess, 1), numTrials);
    pseudolatent_incl(1:nAvail, ii) = thisSess(1:nAvail, 6);  % latent values
end
params.numShuff = nShuffToUse;

fprintf('Extracted latent traces from %d pseudosessions.\n', nShuffToUse);

%% ========================================================================
%  11. SHUFFLE LOOP: ASSESS VARIABLE SIGNIFICANCE
%  ========================================================================
% For each tested variable group, we independently shuffle that variable
% while keeping the rest of the model intact, then completely refit the
% encoding model (RRR + ridge). This is repeated numShuff times to build
% an empirical null distribution. Each shuffle destroys
% the relationship between one specific variable and the neural data, while
% the model retains full dimensionality. By comparing the
% real R^2 to the distribution of shuffled R^2 values, we can assess
% whether each variable contributes significant encoding beyond chance.

varEtest_shuff = zeros(nNeurons, numModels, params.numShuff);

fprintf('\nRunning %d shuffle iterations...\n', params.numShuff);
shuffTimer = tic;

for shuffIdx = 1:params.numShuff
    if mod(shuffIdx, 10) == 0
        elapsed = toc(shuffTimer);
        fprintf('  Shuffle %d/%d (%.1f min elapsed, ~%.1f min remaining)\n', ...
            shuffIdx, params.numShuff, elapsed/60, elapsed/shuffIdx*(params.numShuff-shuffIdx)/60);
    end

    % Generate shuffled predictors for this iteration
    [AstimOutsideLShuff, AstimOutsideRShuff, AstimCenterShuff, ...
     AstimLatentShuff, AchoiceShuff, AlatentMoveShuff] = ...
        genShuffledPredictors(blockTL, inclTrials, params.binSize, expStartStop, ...
                              kernelLags, pseudolatent_incl(:, shuffIdx));

    % Assemble shuffled model set
    modelsShuff = cell(8, 1);
    modelsShuff{1} = AstimOutsideLShuff;
    modelsShuff{2} = AstimOutsideRShuff;
    modelsShuff{3} = AstimCenterShuff;
    modelsShuff{5} = AchoiceShuff;
    modelsShuff{6} = AlatentMoveShuff;
    modelsShuff{7} = AstimLatentShuff;
    for ii = [1:3, 5:7]
        if ~isempty(modelsShuff{ii})
            modelsShuff{ii} = modelsShuff{ii}(:, any(modelsShuff{ii}, 1));
        end
    end

    % Test each variable group
    for m = 1:numModels

        % Rebuild the interaction term (model 8) appropriately:
        %   When shuffling move-context variables, choice stays real.
        %   When shuffling choice, move-context stays real.
        %   When shuffling stim, original interaction is preserved.
        if ismember(m, [4, 5])  % Move-Context or Context: shuffle latent, keep real choice
            modelsShuff{8} = models{5} .* AlatentMoveShuff;
        elseif m == 2  % Choice: shuffle choice, keep real Move-Context
            modelsShuff{8} = AchoiceShuff .* models{6};
        else  % Stim or Stim Context: interaction not in this model group
            modelsShuff{8} = models{8};
        end
        modelsShuff{8} = modelsShuff{8}(:, any(modelsShuff{8}, 1));

        % Replace the target variable group with shuffled versions;
        % all other models remain as the originals.
        modelsForFit = models;
        modelsForFit(modelGroupings{m}) = modelsShuff(modelGroupings{m});

        % Rebuild full predictor matrix and refit RRR
        combinedShuff = [modelsForFit{:}];
        Pshuff = [Atime, AcontextBaseline, combinedShuff(allPredBins,:), ApupilPosition, Aface];
        [~, bShuff, ~, ~] = CanonCor2(allNeurons, Pshuff);
        reducedRanksShuff = Pshuff * bShuff(:, 1:params.numRanks);

        % Determine relevant bins for R^2 computation (same as real model)
        modelSubset = [models{modelGroupings{m}}];
        inclBins = any(modelSubset(allPredBins,:), 2);

        % Fit each neuron with preset lambda from the original fit
        neuronIdx = 0;
        for sn = 1:nShanks
            for jj = 1:numel(cidInclIdxs{sn})
                neuronIdx = neuronIdx + 1;
                B = binnedSpNormSmoothed{sn}(allPredBins, cidInclIdxs{sn}(jj));

                [~, thisR2, ~] = fitRidgeShuffle(reducedRanksShuff, B, ...
                    trainSet, testSet, inclBins, lambdaAll(neuronIdx,:), useGlmnet);
                varEtest_shuff(neuronIdx, m, shuffIdx) = thisR2;
            end
        end
    end
end

fprintf('Shuffle procedure complete (%.1f min).\n', toc(shuffTimer)/60);

%% ========================================================================
%  12. COMPUTE DELTA R^2 AND P-VALUES
%  ========================================================================
% deltaR2 = R2_real - median(R2_shuffled) for each neuron and model.
% p-value: the fraction of shuffles where shuffled R^2 >= real R^2,
% computed as (1 + N - M) / (1 + N) where M is the number of shuffles
% with R^2 < real R^2.

deltaR2 = zeros(nNeurons, numModels);
pValues = zeros(nNeurons, numModels);

for m = 1:numModels
    for nn = 1:nNeurons
        shuffDist = squeeze(varEtest_shuff(nn, m, :));
        deltaR2(nn, m) = varEtest_real(nn, m) - median(shuffDist, 'omitnan');
        M = sum(varEtest_real(nn, m) > shuffDist);
        pValues(nn, m) = (1 + params.numShuff - M) / (1 + params.numShuff);
    end
end

%% ========================================================================
%  13. DISPLAY RESULTS
%  ========================================================================

fprintf('\n============================================================\n');
fprintf('  ENCODING MODEL RESULTS: %s %s (Probe %d)\n', mouseName, dateStr, probeToAnalyze);
fprintf('  %d neurons, %d shuffles\n', nNeurons, params.numShuff);
fprintf('============================================================\n\n');

pThresh = 0.05;
fprintf('%-16s  %8s  %12s  %10s\n', 'Model', 'Sig/Total', 'Med deltaR2', 'Med deltaR2 (sig)');
fprintf('%s\n', repmat('-', 1, 56));
for m = 1:numModels
    nSig = sum(pValues(:,m) < pThresh);
    medAll = median(deltaR2(:,m), 'omitnan');
    if nSig > 0
        medSig = median(deltaR2(pValues(:,m) < pThresh, m), 'omitnan');
    else
        medSig = NaN;
    end
    fprintf('%-16s  %4d/%-4d  %12.4f  %10.4f\n', modelNames{m}, nSig, nNeurons, medAll, medSig);
end

fprintf('\nFull model: median R^2 = %.4f\n', median(varEtest_full, 'omitnan'));

%% ========================================================================
%  14. SINGLE-NEURON DIAGNOSTIC PLOTS
%  ========================================================================
% For a selected neuron, plot:
%   Row 1: Trial-averaged PSTHs split by 4 trial types - real data (shaded
%          SEM) and full model prediction. Left
%          panels are stim-aligned, right panels are move-aligned.
%   Row 2: R^2 null distributions for each tested variable - histogram of
%          shuffled R^2 values with real R^2 marked in red.
%   Row 3: Kernel weights recovered from the RRR projection, split into
%          stim-aligned and move-aligned groups.

% --- Select neuron to plot ---
% Default: pick the neuron with the highest full-model R^2
[~, exNeuronIdx] = max(varEtest_full);
fprintf('\nDiagnostic plot for neuron %d (full model R^2 = %.3f)\n', exNeuronIdx, varEtest_full(exNeuronIdx));

% --- Reconstruct which shank/cidx this global neuron index corresponds to ---
globalIdx = 0;
for sn = 1:nShanks
    for jj = 1:numel(cidInclIdxs{sn})
        globalIdx = globalIdx + 1;
        if globalIdx == exNeuronIdx
            exSn = sn; exLocalIdx = jj; exCidx = cidInclIdxs{sn}(jj);
        end
    end
end

% --- Build PSTH time-sampling matrices ---
% Four trial types defined by block identity and stimulus position:
%   1: Left Block, Outside Left stim    (blockType=-1, stimPos=-1)
%   2: Left Block, Center stim          (blockType=-1, stimPos=+1)
%   3: Right Block, Center stim         (blockType=+1, stimPos=-1)
%   4: Right Block, Outside Right stim  (blockType=+1, stimPos=+1)

btIncl = blockTL.blockType(inclTrials);
spIncl = blockTL.stimPosition(inclTrials);
tg = zeros(numTrials, 1);
tg(btIncl == -1 & spIncl == -1) = 1;
tg(btIncl == -1 & spIncl ==  1) = 2;
tg(btIncl ==  1 & spIncl == -1) = 3;
tg(btIncl ==  1 & spIncl ==  1) = 4;

stimTimes_incl = blockTL.stimulusOnTimes(inclTrials);
moveTimes_incl = blockTL.moveTimes(inclTrials, 1);

% Time axis for the trial bins (used for interpolation)
ttFull = expStartStop(1):params.binSize:expStartStop(2);
ttPred = ttFull(allPredBins);
ttPred = ttPred(1:nT);

% Construct sampling grids for stim-aligned and move-aligned windows
psthBinSize = 0.005;  % 5 ms PSTH resolution
stimWindow = [0 0.15];
moveWindow = [-0.15 0];
stimBins = stimWindow(1):psthBinSize:stimWindow(2);
moveBins = moveWindow(1):psthBinSize:moveWindow(2);

tSampStim = stimTimes_incl + stimBins;  % (nTrials x nBins)
tSampMove = moveTimes_incl + moveBins;

% Mask out stim-aligned bins that fall after movement onset, and
% move-aligned bins that fall before stimulus onset (same logic as kernels)
for ii = 1:numTrials
    tSampStim(ii, tSampStim(ii,:) > moveTimes_incl(ii)) = NaN;
    tSampMove(ii, tSampMove(ii,:) < stimTimes_incl(ii)) = NaN;
end

% Retrieve neural data and model prediction for this neuron
firingClu = binnedSpNormSmoothed{exSn}(allPredBins, exCidx);
predBNeuron = predBAll(:, exNeuronIdx);

% Interpolate onto PSTH grids
psthData_stim = interp1(ttPred, firingClu(1:nT), tSampStim);
psthModel_stim = interp1(ttPred, predBNeuron(1:nT), tSampStim);
psthData_move = interp1(ttPred, firingClu(1:nT), tSampMove);
psthModel_move = interp1(ttPred, predBNeuron(1:nT), tSampMove);

% --- Trial type colors ---
% Left Block Outside = dark blue, Left Block Center = dark orange,
% Right Block Center = light blue, Right Block Outside = light orange
colorsToPlot =  [0 0.25 0.5; 0.5 0.25 0; 0 0.5 1; 1 0.5 0];
lineWidths = [1 2 2 1];
tgLabels = {'LB Outside L', 'LB Center', 'RB Center', 'RB Outside R'};

% ===== FIGURE =====
nCol = max(numModels, 4);
fig = figure('Position', [50 50 1600 900], 'Color', 'w');
sgtitle(sprintf('%s  %s  - Shank %d Cluster %d  (%s)', ...
    mouseName, dateStr, exSn-1, sp{pn}(exSn).cids(exCidx), sp{pn}(exSn).cidRegLabels{exCidx}), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

% ----- Row 1: PSTHs -----
% Panel 1: Neuron PSTH (stim-aligned)
ax1 = subplot(3, nCol, 1);
plotPSTH(ax1, psthData_stim, stimBins, tg, colorsToPlot, lineWidths, true);
title(ax1, 'Neuron PSTH'); ylabel(ax1, 'sp/s (norm)');
xlabel(ax1, 'Time from stim (s)');

% Panel 2: Full model PSTH (stim-aligned)
ax2 = subplot(3, nCol, 2);
plotPSTH(ax2, psthModel_stim, stimBins, tg, colorsToPlot, lineWidths, false);
title(ax2, sprintf('Full Model (R^2 = %.3f)', varEtest_full(exNeuronIdx)));
xlabel(ax2, 'Time from stim (s)');

% Panel 3: Neuron PSTH (move-aligned)
ax3 = subplot(3, nCol, 3);
plotPSTH(ax3, psthData_move, moveBins, tg, colorsToPlot, lineWidths, true);
title(ax3, 'Neuron PSTH'); xlabel(ax3, 'Time from move (s)');

% Panel 4: Full model PSTH (move-aligned)
ax4 = subplot(3, nCol, 4);
plotPSTH(ax4, psthModel_move, moveBins, tg, colorsToPlot, lineWidths, false);
title(ax4, 'Full Model (move-aligned)');
xlabel(ax4, 'Time from move (s)');
% legend(ax4, tgLabels, 'Location', 'east', 'FontSize', 7);

linkaxes([ax1 ax2], 'xy'); linkaxes([ax3 ax4], 'xy');
yLimStim = get(ax1, 'YLim'); yLimMove = get(ax3, 'YLim');
yLimAll = [0, max([yLimStim(2), yLimMove(2)])];
ylim([ax1 ax2 ax3 ax4], yLimAll);

% ----- Row 2: R^2 null distributions -----
for m = 1:numModels
    axH = subplot(3, nCol, nCol + m);
    shuffDist = squeeze(varEtest_shuff(exNeuronIdx, m, :));
    histogram(axH, shuffDist, 30, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); hold on;
    xline(axH, varEtest_real(exNeuronIdx, m), '-r', 'LineWidth', 2);
    title(axH, sprintf('%s (p=%.3f)', modelNames{m}, pValues(exNeuronIdx, m)));
    if m == 1; ylabel(axH, '# Shuffles'); end
    xlabel(axH, 'R^2');
    set(axH, 'Box', 'off', 'TickDir', 'out');
end

% ----- Row 3: Kernel weights -----
% Recover predictor-space weights from the RRR coefficients. For K-fold CV,
% average coefficients across folds (excluding the intercept in row 1).
xMean = mean(xAll{exNeuronIdx}(2:end, :), 2);  % (numRanks x 1)
kernelWeights = bComb(:, 1:params.numRanks) * xMean;  % (nPredictors_full x 1)

% Extract task model kernels (skip Atime and AcontextBaseline columns)
allKernels = cell(numel(models), 1);
for ii = 1:numel(models)
    allKernels{ii} = kernelWeights(nNuisanceCols + modelIndices{ii});
end

% Stim-aligned kernels: models 1 (OutsideL), 2 (OutsideR), 3 (Center), 7 (Context-stim)
visKernelIdxs = [1 2 3 7];
axK1 = subplot(3, nCol, 2*nCol + 1);
for kk = 1:numel(visKernelIdxs)
    ki = visKernelIdxs(kk);
    kLen = numel(allKernels{ki});
    tAxis = kernelLags{1}(1:kLen) * params.binSize;
    plot(axK1, tAxis, allKernels{ki}, '-', 'LineWidth', 1.25); hold on;
end
legend(axK1, individualModelNames(visKernelIdxs), 'Location', 'northoutside', 'FontSize', 7);
xlabel(axK1, 'Time from stim (s)'); ylabel(axK1, 'Kernel weight');
xlim(axK1, stimWindow); box(axK1, 'off'); set(axK1, 'TickDir', 'out');

% Move-aligned kernels: models 4 (Movement), 5 (Choice), 6 (Context-move), 8 (ChoicexContext)
moveKernelIdxs = [4 5 6 8];
axK2 = subplot(3, nCol, 2*nCol + 2);
for kk = 1:numel(moveKernelIdxs)
    ki = moveKernelIdxs(kk);
    kLen = numel(allKernels{ki});
    tAxis = kernelLags{2}(end-kLen+1:end) * params.binSize;
    plot(axK2, tAxis, allKernels{ki}, '-', 'LineWidth', 1.25); hold on;
end
legend(axK2, individualModelNames(moveKernelIdxs), 'Location', 'northoutside', 'FontSize', 7);
xlabel(axK2, 'Time from move (s)');
xlim(axK2, moveWindow); box(axK2, 'off'); set(axK2, 'TickDir', 'out');

% Panel 3 of row 3: deltaR2 bar plot for this neuron
axB = subplot(3, nCol, 2*nCol + 3);
barColors = zeros(numModels, 3);
barColors(pValues(exNeuronIdx,:) < pThresh, :) = repmat([0.8 0.2 0.2], sum(pValues(exNeuronIdx,:) < pThresh), 1);
barColors(pValues(exNeuronIdx,:) >= pThresh, :) = repmat([0.6 0.6 0.6], sum(pValues(exNeuronIdx,:) >= pThresh), 1);
bh = bar(axB, deltaR2(exNeuronIdx, :)); hold on;
bh.FaceColor = 'flat'; bh.CData = barColors;
set(axB, 'XTickLabel', modelNames, 'XTickLabelRotation', 45, 'Box', 'off', 'TickDir', 'out');
ylabel(axB, '\DeltaR^2'); title(axB, 'Encoding strength');

fprintf('\nDone.\n');

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function plotPSTH(ax, psthData, bins, tg, colors, lineWidths, showSEM)
% PLOTPSTH  Plot trial-averaged PSTHs split by trial type with optional SEM shading.
    for tgIdx = 1:4
        theseTrials = psthData(tg == tgIdx, :);
        mu = mean(theseTrials, 1, 'omitnan');
        plot(ax, bins, mu, 'Color', colors(tgIdx,:), 'LineWidth', lineWidths(tgIdx)); hold(ax, 'on');
        if showSEM && size(theseTrials, 1) > 1
            sem = std(theseTrials, [], 1, 'omitnan') / sqrt(size(theseTrials, 1));
            patch(ax, [bins fliplr(bins)], [mu-sem fliplr(mu+sem)], ...
                  colors(tgIdx,:), 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        end
    end
    set(ax, 'Box', 'off', 'TickDir', 'out');
end