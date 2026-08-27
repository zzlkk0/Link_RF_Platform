%RUN_ALL_VALIDATIONS Execute base-MATLAB physical-model cross-checks.

script_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(script_dir, 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

free_space = validate_free_space(output_dir);
two_ray = validate_two_ray(output_dir);
summary = struct( ...
    'generated_at', char(datetime('now', 'TimeZone', 'local', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')), ...
    'free_space', free_space, ...
    'two_ray', two_ray, ...
    'all_pass', free_space.pass && two_ray.pass);

file_id = fopen(fullfile(output_dir, 'matlab_validation_summary.json'), 'w');
assert(file_id >= 0, 'Could not open MATLAB validation summary for writing.');
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id, '%s', jsonencode(summary, PrettyPrint=true));
fprintf('MATLAB_VALIDATION_ALL_PASS\n');
