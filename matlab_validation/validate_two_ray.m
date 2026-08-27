function result = validate_two_ray(output_dir)
%VALIDATE_TWO_RAY Verify fading and nonlinear phase without extra toolboxes.

if nargin < 1
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'output');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

c0 = 299792458;
f = (2.402e9:1.0e6:2.480e9).';
horizontal_m = 8.0;
tx_height_m = 1.3;
rx_height_m = 0.9;
direct_m = hypot(horizontal_m, tx_height_m-rx_height_m);
reflected_m = hypot(horizontal_m, tx_height_m+rx_height_m);
reflection = -0.72 + 0.08j;

h_direct = c0 ./ (4*pi*f*direct_m) .* exp(-1j*2*pi*f*direct_m/c0);
h_reflected = reflection*c0 ./ (4*pi*f*reflected_m) ...
    .* exp(-1j*2*pi*f*reflected_m/c0);
h_total = h_direct + h_reflected;

amplitude_db = 20*log10(abs(h_total));
phase = unwrap(angle(h_total));
centered_frequency = f - mean(f);
fit_coefficients = polyfit(centered_frequency, phase, 1);
phase_fit = polyval(fit_coefficients, centered_frequency);
phase_residual_rms_rad = sqrt(mean((phase-phase_fit).^2));
amplitude_span_db = max(amplitude_db)-min(amplitude_db);
assert(amplitude_span_db > 0.2, 'Two-ray fading was not observed.');
assert(phase_residual_rms_rad > 1e-4, 'Two-ray phase remained unexpectedly linear.');

result = struct( ...
    'test', 'two_ray', ...
    'amplitude_span_db', amplitude_span_db, ...
    'phase_residual_rms_rad', phase_residual_rms_rad, ...
    'pass', true);
fprintf('MATLAB_TWO_RAY_PASS span=%.3f dB phase_rms=%.5f rad\n', ...
    amplitude_span_db, phase_residual_rms_rad);
end
