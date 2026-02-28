function shuffled_vector = shuffle_sequences(input_vector)
% SHUFFLE_SEQUENCES  Shuffle a binary sequence while preserving run-length structure.
%   Separates the input into runs of each value, independently permutes the
%   runs, and interleaves them. This preserves the autocorrelation structure
%   (i.e., the distribution of run lengths) while destroying the trial-by-trial
%   relationship to other variables.

    u_vals = unique(input_vector);
    if length(u_vals) ~= 2
        error('Input vector must contain exactly two unique values.');
    end
    val1 = u_vals(1);
    val2 = u_vals(2);

    % Separate into runs
    seqs_val1 = {};
    seqs_val2 = {};
    current_sequence = input_vector(1);

    for i = 2:length(input_vector)
        if input_vector(i) == current_sequence(end)
            current_sequence = [current_sequence, input_vector(i)];
        else
            if current_sequence(1) == val1
                seqs_val1 = [seqs_val1, {current_sequence}];
            else
                seqs_val2 = [seqs_val2, {current_sequence}];
            end
            current_sequence = input_vector(i);
        end
    end
    if current_sequence(1) == val1
        seqs_val1 = [seqs_val1, {current_sequence}];
    else
        seqs_val2 = [seqs_val2, {current_sequence}];
    end

    % Shuffle runs independently
    seqs_val1 = seqs_val1(randperm(length(seqs_val1)));
    seqs_val2 = seqs_val2(randperm(length(seqs_val2)));

    % Determine start order
    if length(seqs_val1) == length(seqs_val2)
        start_with_val1 = (rand() > 0.5);
    else
        start_with_val1 = length(seqs_val1) > length(seqs_val2);
    end

    % Interleave
    shuffled_vector = [];
    max_len = max(length(seqs_val1), length(seqs_val2));
    for i = 1:max_len
        if start_with_val1
            if i <= length(seqs_val1), shuffled_vector = [shuffled_vector, seqs_val1{i}]; end
            if i <= length(seqs_val2), shuffled_vector = [shuffled_vector, seqs_val2{i}]; end
        else
            if i <= length(seqs_val2), shuffled_vector = [shuffled_vector, seqs_val2{i}]; end
            if i <= length(seqs_val1), shuffled_vector = [shuffled_vector, seqs_val1{i}]; end
        end
    end
    shuffled_vector = shuffled_vector';
end
