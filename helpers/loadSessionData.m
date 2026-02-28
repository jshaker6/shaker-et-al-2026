function [blockTL, sp, motSVD, pupilCoords] = loadSessionData(dataRoot, mouseName, dateStr, probeNames)
% LOADSESSIONDATA  Load all data for one recording session from FigShare structure.
%   Reads behavioral data (blockTL), spike sorting outputs, quality metrics,
%   face motion SVD, and pupil coordinates from the ShakerEtAl_2026 folder.
%
%   dataRoot   - path to ShakerEtAl_2026 root folder
%   mouseName  - e.g., 'JRS_0028'
%   dateStr    - e.g., '2023-08-07'
%   probeNames - cell array of probe name strings, e.g., {'ProbeA', 'ProbeB'}
%
%   blockTL     - struct with behavioral data and latent model outputs
%   sp          - cell array {nProbes x 1}, each containing struct array (1 x 4)
%                 with fields: .st, .clu, .cids, .refracOK, .spikeFloorOK,
%                 .cidCoords, .cidRegInds, .cidRegLabels
%   motSVD      - (nFrames x 16) face motion SVD components, or [] if missing
%   pupilCoords - (nFrames x 4) pupil coordinates, or [] if missing

    sessDir = fullfile(dataRoot, mouseName, dateStr);
    nShanks = 4;

    %% Behavioral data
    tmp = load(fullfile(sessDir, 'blockTL.mat'));
    fnames = fieldnames(tmp);
    blockTL = tmp.(fnames{1});

    %% Spike sorting outputs and quality metrics
    nProbes = numel(probeNames);
    sp = cell(nProbes, 1);

    for pn = 1:nProbes
        for sn = 0:(nShanks - 1)
            shankDir = fullfile(sessDir, probeNames{pn}, sprintf('shank%d', sn));
            if ~isfolder(shankDir)
                sp{pn}(sn+1).st = [];
                sp{pn}(sn+1).clu = [];
                sp{pn}(sn+1).cids = [];
                continue;
            end

            % Spike times and cluster IDs
            sp{pn}(sn+1).st = readNPY(fullfile(shankDir, 'spike_times.npy'));
            sp{pn}(sn+1).clu = readNPY(fullfile(shankDir, 'spike_clusters.npy'));
            sp{pn}(sn+1).cids = readNPY(fullfile(shankDir, 'cids.npy'));

            % Quality metrics
            if isfile(fullfile(shankDir, 'refracOK.npy'))
                sp{pn}(sn+1).refracOK = readNPY(fullfile(shankDir, 'refracOK.npy'));
            else
                sp{pn}(sn+1).refracOK = nan(numel(sp{pn}(sn+1).cids), 1);
            end
            if isfile(fullfile(shankDir, 'spikeFloorOK.npy'))
                sp{pn}(sn+1).spikeFloorOK = readNPY(fullfile(shankDir, 'spikeFloorOK.npy'));
            else
                sp{pn}(sn+1).spikeFloorOK = nan(numel(sp{pn}(sn+1).cids), 1);
            end
            
            % Anatomical registration
            if isfile(fullfile(shankDir, 'cidCoords.csv'))
                sp{pn}(sn+1).cidCoords = readmatrix(fullfile(shankDir, 'cidCoords.csv'));
            else
                sp{pn}(sn+1).cidCoords = nan(numel(sp{pn}(sn+1).cids), 3);
            end
            if isfile(fullfile(shankDir, 'cidRegInds.csv'))
                sp{pn}(sn+1).cidRegInds = readmatrix(fullfile(shankDir, 'cidRegInds.csv'));
            else
                sp{pn}(sn+1).cidRegInds = nan(numel(sp{pn}(sn+1).cids), 1);
            end
            if isfile(fullfile(shankDir, 'cidRegLabels.csv'))
                sp{pn}(sn+1).cidRegLabels = readcell(fullfile(shankDir, 'cidRegLabels.csv'),'FileType','spreadsheet');
            else
                sp{pn}(sn+1).cidRegLabels = nan(numel(sp{pn}(sn+1).cids), 1);
            end
        end
    end

    %% Face motion SVD
    motFile = fullfile(sessDir, 'motSVD.npy');
    if isfile(motFile)
        motSVD = readNPY(motFile);
    else
        motSVD = [];
    end

    %% Pupil coordinates
    pupilFile = fullfile(sessDir, 'pupilCoords.npy');
    if isfile(pupilFile)
        pupilCoords = readNPY(pupilFile);
    else
        pupilCoords = [];
    end
end
