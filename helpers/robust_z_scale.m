function scaled = robust_z_scale(data, varargin)
% ROBUST_Z_SCALE  Robust z-scoring using median and MAD.
%   Centers data to median 0 and scales by the median absolute deviation
%   multiplied by the consistency constant (1.4826), producing units
%   equivalent to standard z-scores under normality.
%
%   scaled = robust_z_scale(data)
%   scaled = robust_z_scale(data, 'clip_range', [-10 10])
%   scaled = robust_z_scale(data, 'outlier_handling', 'remove')
%
%   data   - input matrix (operates over all elements jointly)
%   scaled - robust z-scored output (same size as data)

    p = inputParser;
    addParameter(p, 'clip_range', [], @isnumeric);
    addParameter(p, 'outlier_handling', 'none', @ischar);
    parse(p, varargin{:});

    allVals = data(:);
    med = median(allVals, 'omitnan');
    madVal = median(abs(allVals - med), 'omitnan');
    scale = 1.4826 * madVal;

    if scale == 0
        scaled = zeros(size(data));
        return;
    end

    scaled = (data - med) / scale;

    if ~isempty(p.Results.clip_range)
        scaled = max(min(scaled, p.Results.clip_range(2)), p.Results.clip_range(1));
    end

    if strcmp(p.Results.outlier_handling, 'remove')
        scaled(abs(scaled) > 10) = NaN;
    end
end
