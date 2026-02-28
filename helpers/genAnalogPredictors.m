function [ApupilPosition, Aface] = genAnalogPredictors(tDigitalUp, pupilCoords, motSVD, nPCs, binSize, expStartStop)
% GENANALOGPREDICTORS  Construct face motion and pupil position predictor matrices.
%   Interpolates camera-sampled face motion SVD components and pupil position
%   onto the time bins used for neural activity, producing instantaneous
%   (non-lagged) analog predictors.
%
%   tDigitalUp  - (nFrames x 1) camera trigger timestamps on probe clock
%   pupilCoords - (nFrames x 4) pupil landmark coordinates from FigShare:
%                 [dorsal_y, ventral_y, medial_x, lateral_x]
%   motSVD      - (nFrames x nComponents) face motion energy SVD components
%   nPCs        - number of SVD components to use (default 16)
%   binSize     - time bin width in seconds
%   expStartStop - [start, end] times of experiment
%
%   ApupilPosition - (nTimeBins x 2) pupil center [x, y] interpolated to bins
%   Aface          - (nTimeBins x nPCs) face motion SVD interpolated to bins

    tt = expStartStop(1):binSize:expStartStop(2);
    frameIncl = ~isnan(tDigitalUp);

    % Pupil center of mass from DLC landmark coordinates
    pupilCoords = double(pupilCoords);
    pupilExp = interp1(tDigitalUp(frameIncl), pupilCoords(frameIncl, :), tt);
    ApupilPosition = zeros(length(tt), 2);
    ApupilPosition(:, 1) = (pupilExp(:, 3) + pupilExp(:, 4)) / 2;  % x: mean of medial and lateral
    ApupilPosition(:, 2) = (pupilExp(:, 1) + pupilExp(:, 2)) / 2;  % y: mean of dorsal and ventral
    ApupilPosition = fillmissing(ApupilPosition, 'linear', 1);

    % Face motion SVD components
    motSVD = double(motSVD);
    Aface = interp1(tDigitalUp(frameIncl), motSVD(frameIncl, 1:nPCs), tt);
end
