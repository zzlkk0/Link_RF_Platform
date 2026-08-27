function result = validate_free_space(output_dir)
%VALIDATE_FREE_SPACE Cross-check the Python Level-0 equations in base MATLAB.

if nargin < 1
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'output');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

c0 = 299792458;
distance_m = 8.0;
frequencies_hz = (2.402e9:1.0e6:2.480e9).';
response = c0 ./ (4*pi*frequencies_hz*distance_m) ...
    .* exp(-1j*2*pi*frequencies_hz*distance_m/c0);

centered_frequency = frequencies_hz - mean(frequencies_hz);
fit_coefficients = polyfit(centered_frequency, unwrap(angle(response)), 1);
distance_estimate_m = -c0*fit_coefficients(1)/(2*pi);
error_m = distance_estimate_m - distance_m;
assert(abs(error_m) < 1e-8, 'Free-space distance recovery failed.');

result = struct( ...
    'test', 'free_space', ...
    'true_distance_m', distance_m, ...
    'estimated_distance_m', distance_estimate_m, ...
    'error_m', error_m, ...
    'pass', true);
fprintf('MATLAB_FREE_SPACE_PASS distance=%.9f m error=%.3g m\n', ...
    distance_estimate_m, error_m);
end
