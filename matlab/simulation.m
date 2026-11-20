
clear; clc; close all;

%% 1. Radar parameters
c = 3e8;             % Speed of light (m/s)
B = 60e6;             % LFM bandwidth (Hz)
Tp = 8e-6;             % Pulse width (s)
fs = 120e6;            % Sampling frequency (Hz)
PRI = 100e-6;          % Pulse repetition interval (s)
fc = 10e9;             % Example carrier frequency (Hz)

K = B / Tp;            % Chirp slope
lambda = c / fc;       % Wavelength

%% 2. Scene model
radarPos = [0 0];      % Radar position (m)

% Three target positions (m)
targetPos = [
    3000  0
    6000  0
    9000  0
];

% Calculate target ranges
R = sqrt((targetPos(:,1)-radarPos(1)).^2 + ...
         (targetPos(:,2)-radarPos(2)).^2).';

% Example radial velocities and RCS
v = [50 -30 20];       % Velocity (m/s)
RCS = [1 8 3];          % RCS (m^2)

tau = 2 * R / c;        % Round-trip delay
fd = 2 * v / lambda;    % Doppler frequency (Hz)

%% 3. Generate the transmitted LFM pulse
t = 0 : 1/fs : PRI-1/fs;

tx = zeros(size(t));
pulse = (t >= 0) & (t < Tp);

tx(pulse) = exp(1j*pi*K*t(pulse).^2);

%% 4. Simulate the received echoes
% Simplified relative echo amplitude:
% voltage amplitude is proportional to sqrt(RCS)/R^2
amp = sqrt(RCS) ./ (R/1000).^2;
amp = amp / max(amp);

rx = zeros(size(t));

for k = 1:length(R)

    % Time relative to the arrival of target k's echo
    td = t - tau(k);

    % Echo exists only during the delayed pulse
    idx = (td >= 0) & (td < Tp);

    echo = amp(k) ...
        .* exp(1j*pi*K*td(idx).^2) ...
        .* exp(1j*2*pi*fd(k)*t(idx)) ...
        .* exp(-1j*2*pi*fc*tau(k));

    rx(idx) = rx(idx) + echo;
end

% Add complex Gaussian noise
noiseAmp = 0.005;
rx = rx + noiseAmp/sqrt(2) * ...
    (randn(size(rx)) + 1j*randn(size(rx)));

%% 5. Display the radar scene
figure('Name','Radar Scene');

plot([0 max(R)/1000], [0 0], 'k-', ...
    'LineWidth', 1.5);
hold on;

plot(0, 0, 'kp', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'k');

plot(R/1000, zeros(size(R)), 'ro', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r');

for k = 1:length(R)
    text(R(k)/1000, 0.04, ...
        sprintf('T%d: %.0f km', k, R(k)/1000), ...
        'HorizontalAlignment', 'center');
end

xlabel('Range (km)');
ylabel('Scene');
title('One-dimensional Radar Scene');
xlim([-0.5 10]);
ylim([-0.1 0.15]);
yticks([]);
grid on;

%% 6. Plot transmitted signal and received echoes
figure('Name','LFM Signal and Echoes');

subplot(2,1,1);
plot(t(pulse)*1e6, real(tx(pulse)));
xlabel('Time (us)');
ylabel('Amplitude');
title('Transmitted LFM Signal');
grid on;

subplot(2,1,2);
plot(t*1e6, abs(rx));
xlabel('Time (us)');
ylabel('Magnitude');
title('Received Echoes from Three Targets');
xlim([0 70]);
grid on;
hold on;

for k = 1:length(R)
    xline(tau(k)*1e6, '--', ...
        sprintf('T%d: %.0f km', k, R(k)/1000));
end

%% 7. Print important parameters
fprintf('Range resolution: %.2f m\n', c/(2*B));
fprintf('Maximum unambiguous range: %.1f km\n', ...
    c*PRI/2/1000);

for k = 1:length(R)
    fprintf(['Target %d: R = %.1f km, delay = %.1f us, ' ...
        'RCS = %.1f m^2, velocity = %.1f m/s\n'], ...
        k, R(k)/1000, tau(k)*1e6, RCS(k), v(k));
end





%% 8. Pulse Compression and Automatic Target Detection

% 1. Matched filtering
txPulse = tx(pulse);
mfFilter = conj(fliplr(txPulse));
mfOut = conv(rx, mfFilter, 'full');

% 2. Convert time to range
Npulse = length(txPulse);
tMF = ((0:length(mfOut)-1) - (Npulse-1)) / fs;
rangeMF = c * tMF / 2;
rangeKm = rangeMF / 1000;

% 3. Keep the range of interest (0-12 km)
valid = rangeKm >= 0 & rangeKm <= 12;
rangePlot = rangeKm(valid);
magPlot = abs(mfOut(valid));

% 4. Normalize the range profile in dB
mfDBPlot = 20*log10( ...
    magPlot / (max(magPlot) + eps) + 1e-12);

% 5. Automatic target detection
[pks, locs] = findpeaks(mfDBPlot, rangePlot, ...
    'MinPeakHeight', -25, ...
    'MinPeakDistance', 0.1, ...
    'MinPeakProminence', 5);

% Sort detected targets by range
[locs, order] = sort(locs);
pks = pks(order);

% 6. Print detection results
fprintf('\n===== Target Detection Results =====\n');

if isempty(locs)
    fprintf('No targets detected. Check the threshold.\n');
else
    for k = 1:length(locs)
        fprintf('Target %d: Range = %.3f km, Peak = %.2f dB\n', ...
            k, locs(k), pks(k));
    end
end

% 7. Display pulse compression and detection in ONE figure
figure('Name', 'Pulse Compression and Target Detection');

plot(rangePlot, mfDBPlot, 'b', 'LineWidth', 1.2);
hold on;

% Detected target peaks
plot(locs, pks, 'rv', ...
    'MarkerFaceColor', 'r', 'MarkerSize', 8);

% Label each detected target
for k = 1:length(locs)
    text(locs(k), pks(k), ...
        sprintf('  %.3f km', locs(k)), ...
        'Color', 'r', ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'left');
end

% Detection threshold
yline(-25, '--k', 'Detection Threshold (-25 dB)');

xlabel('Range (km)');
ylabel('Normalized Magnitude (dB)');
title('Pulse Compression and Automatic Target Detection');

xlim([0 12]);
ylim([-60 5]);
grid on;

legend('Pulse-compressed signal', ...
       'Detected targets', ...
       'Detection threshold', ...
       'Location', 'southwest');
