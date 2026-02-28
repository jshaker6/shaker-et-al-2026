function [trainSet, testSet, allPredBins, nT, trialNums] = createCVsplit_KFold(expStartStop, binSize, stimOnTimes, moveTimes, kernelLags, numFolds)
% CREATECVSPLIT_KFOLD  Create K-fold cross-validation splits at the trial level.
%   Assigns time bins to trials based on the interval from stimulus onset to
%   movement onset (extended by the movement kernel lag), then partitions
%   trials into K folds. Returns logical index vectors over the concatenated
%   trial-bin space (excluding inter-trial intervals).
%
%   expStartStop - [start, end] times of experiment
%   stimOnTimes  - (nTrials x 1) stimulus onset times
%   moveTimes    - (nTrials x 1) movement onset times
%   kernelLags   - {2 x 1} cell: {1} = stim kernel lag indices, {2} = move kernel lag indices
%   numFolds     - number of CV folds
%
%   trainSet  - cell array {numFolds x 1}, each a logical vector over trial bins
%   testSet   - cell array {numFolds x 1}, each a logical vector over trial bins
%   allPredBins - logical vector over full time axis indicating bins within any trial (i.e., bins to be predicted by the model)
%   nT        - total number of trial bins
%   trialNums - (nT x 1) trial identity for each bin

    tt = expStartStop(1):binSize:expStartStop(2);
    nTFull = numel(tt) - 1;

    % Map each time bin to a trial number
    trialIntervals = [stimOnTimes, moveTimes] - expStartStop(1);
    trialNumsFull = zeros(nTFull, 1);
    for ii = 1:size(trialIntervals, 1)
        bins = round(trialIntervals(ii,1)/binSize):round(trialIntervals(ii,2)/binSize) + kernelLags{2}(end);
        bins = bins(bins >= 1 & bins <= nTFull);
        trialNumsFull(bins) = ii;
    end

    % Partition trials into folds
    trialSize = size(trialIntervals, 1);
    cvPartition = cvpartition(trialSize, 'Kfold', numFolds);

    trainSet = cell(numFolds, 1);
    testSet = cell(numFolds, 1);
    for ii = 1:numFolds
        trainTrials = find(training(cvPartition, ii));
        testTrials = find(test(cvPartition, ii));
        trainSetFull = ismember(trialNumsFull, trainTrials);
        testSetFull = ismember(trialNumsFull, testTrials);

        xyz = zeros(nTFull, 1);
        xyz(trainSetFull) = 1;
        xyz(testSetFull) = 2;
        xyz(~trainSetFull & ~testSetFull) = [];
        nT = length(xyz);
        trainSet{ii} = (xyz == 1);
        testSet{ii} = (xyz == 2);
    end

    allPredBins = (trialNumsFull > 0);
    trialNums = trialNumsFull(allPredBins);
end
