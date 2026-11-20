% radar_RD_RDA_HarmonyOS_easy.m
% Eight-element array, genuine range/velocity/azimuth estimation from echoes.
% 2-D CA-CFAR on selectable 8Rx noncoherent power, then DBF azimuth
% on 8 complex array channels at each detected RD peak.
% Range-Doppler-Azimuth (RDA) is not XYZ: x=R*cos(az), y=R*sin(az).
% Paired weather evaluation uses the existing 'original' profile.
% All weathers share target truth, receiver noise and detector settings.
% Zero-Doppler notch is selectable, and OFF for the easier demonstration.
% Simplified far-field narrowband array model; no physical RF hardware.
% Original radar simulation + traditional 2-D CA-CFAR + HTTP dashboard uploader
% Radar coursework: pulse-Doppler LFM data generation, weather and 2-D CFAR
% Output: KuRALS-PD-like 800(range) x 64(Doppler) magnitude maps.
% IMPORTANT: not a hardware-identical KuRALS emulator; no dual-pulse fusion,
% transmit beam scanning, measured RCS, or calibrated rain-scattering model.
% MATLAB R2019b+; base MATLAB only (no findpeaks/Deep Learning toolbox needed).
clear; clc; close all;
rng(2026,'twister');

%% 1. Configuration
% 100 paired scenes = 300 frames. Use 25 scenes for a shorter 75-frame check.
numScenes = 100;

%% Dashboard streaming configuration (Python server must be started first)
enableDashboardUpload = strcmp(getenv('RADAR_DASHBOARD_UPLOAD'), '1');
dashboardServerURL = 'http://127.0.0.1:8765/api/frame';
uploadEveryNFrames = 1;       % Upload every RD frame; increase to reduce network traffic.
uploadPauseSeconds = 1.0;      % Playback delay after each successfully uploaded frame.
dashboardUploadStopped = false; % Turn off uploads after first network error, not radar simulation.

difficultyProfile = 'original';   % Existing profile selector; use 'demo_easy' for an easy demo.
% Detector modes:
% 'array_noncoherent' uses all 8 receivers in the CFAR power map.
% 'single_rx' reproduces the original first-receiver detector.
detectionPowerMode = 'array_noncoherent';
% OFF keeps slow targets visible. Turn ON for the old KuRALS-like notch.
enableZeroDopplerNotch = false;
% Use multiples of 25 scenes to balance all five SNR levels and empty scenes.
weatherList = {'clear','rain','fog'};

switch lower(difficultyProfile)
    case 'demo_easy'
        % Dedicated EASY demonstration scene: target detection from echoes,
        % NOT ground-truth injection, and NOT a claimed algorithm accuracy gain.
        referenceSNRLevels_dB = [0 5 10 15 20];
        pfaNominal = 1e-6;
        rainRateMin_mmph = 2;
        rainRateMax_mmph = 8;
        rainScatterers = 6;
        rainClutterAmpMin = 0.001;
        rainClutterAmpMax = 0.004;
        fogLWCMin_gm3 = 0.03;
        fogLWCMax_gm3 = 0.12;
        sceneRangeMin_m = 800;
        sceneRangeMax_m = 3800;
        rcsMin_m2 = 0.8;
        rcsMax_m2 = 2.5;
    case 'reduced'
        % Less extreme weak-target cases (+8 dB versus baseline).
        referenceSNRLevels_dB = [-15 -10 -5 0 5];
        pfaNominal = 1e-5;
        % Light-to-moderate rain rather than allowing heavy rain.
        rainRateMin_mmph = 2;
        rainRateMax_mmph = 12;
        rainScatterers = 12;
        rainClutterAmpMin = 0.002;
        rainClutterAmpMax = 0.008;
        % Light-to-moderate fog. This changes attenuation only, not fog clutter.
        fogLWCMin_gm3 = 0.03;
        fogLWCMax_gm3 = 0.20;
        sceneRangeMin_m = 600;
        sceneRangeMax_m = 5600;
        rcsMin_m2 = 10^(-0.7);
        rcsMax_m2 = 10^(0.5);
    case 'original'
        % Paired weather comparison across weak-to-strong target conditions.
        referenceSNRLevels_dB = [-20 -15 -10 -5 0];
        pfaNominal = 1e-6;
        rainRateMin_mmph = 2;
        rainRateMax_mmph = 30;
        rainScatterers = 40;
        rainClutterAmpMin = 0.009;
        rainClutterAmpMax = 0.037;
        fogLWCMin_gm3 = 0.05;
        fogLWCMax_gm3 = 5;
        sceneRangeMin_m = 800;
        sceneRangeMax_m = 5600;
        rcsMin_m2 = 10^(-0.7);
        rcsMax_m2 = 10^(0.5);
    otherwise
        error('Unknown difficultyProfile: %s',difficultyProfile);
end

% A distinct folder avoids overwriting the previous fixed3000 dataset.
bundleRoot = fileparts(fileparts(mfilename('fullpath')));
outputDir = fullfile(bundleRoot, 'outputs', 'array_sim', sprintf('radar_dataset_RDA8_%s_%s_%d_HarmonyOS_%s', ...
    difficultyProfile, detectionPowerMode, numScenes*numel(weatherList), ...
    datestr(now,'yyyymmdd_HHMMSS')));
if ~exist(outputDir,'dir'), mkdir(outputDir); end
fprintf('Profile=%s | scenes=%d | frames=%d | folder=%s\n', ...
    difficultyProfile,numScenes,numScenes*numel(weatherList),outputDir);

c = 3e8;
fc = 16e9;                        % KuRALS-PD approximate carrier
lambda = c/fc;
% Eight-receiver uniform linear array along local y direction.
% Azimuth is measured from forward x-axis towards positive y-axis.
numRx = 8;
rxSpacing_m = lambda/2;
rxIndex = (0:numRx-1).';
azimuthSearch_deg = -60:0.5:60;
steeringMatrix = exp(1j*2*pi*(rxSpacing_m/lambda)* ...
    rxIndex*sind(azimuthSearch_deg)); % 8 x numAngles
matchAzTolerance_deg = 12;       % Evaluation-only angular matching tolerance.
B = 20e6;                         % 7.5 m theoretical range resolution
fs = 40e6;                        % 3.75 m fast-time sample spacing
Tp = 2e-6;                        % illustrative single pulse width
K = B/Tp;
numPulses = 64;
numRangeBins = 800;
rangeStep = 7.5;
velStep = 1.769;
% Choose PRI consistent with 64 Doppler bins x 1.769 m/s.
% This is 82.806 us, close to the 83.3 us reported by KuRALS-PD.
PRI = lambda/(2*numPulses*velStep);
Nfast = round(PRI*fs);
Np = round(Tp*fs);
tFast = (0:Nfast-1)'/fs;
tPulse = (0:Np-1)'/fs;
pulseIDs = 0:numPulses-1;

% Symmetric complex-baseband chirp: instantaneous f ~ [-B/2, +B/2]
txPulse = exp(1j*pi*K*(tPulse - Tp/2).^2);
% 快时间加权，抑制脉冲压缩后的距离旁瓣
nFastWin = (0:Np-1).';
wRange = 0.54 - 0.46*cos(2*pi*nFastWin/(Np-1));
mfFilter = conj(flipud(txPulse)) .* wRange;
fftLen = 2^nextpow2(Nfast+Np-1);
H = single(fft(mfFilter,fftLen,1)); % FFT convolution per receiver (single precision)
% Full convolution peak for an echo delayed by d samples: index Np+d
rangeSampleIndex = Np + (0:numRangeBins-1)*2;
assert(max(rangeSampleIndex)<=fftLen, 'Insufficient fast-time samples');
rangeAxis = (0:numRangeBins-1)'*rangeStep;
velAxis = ((0:numPulses-1) - numPulses/2)*velStep;
dopplerWindow = 0.5-0.5*cos(2*pi*pulseIDs/(numPulses-1));

% Signal level: input SNR of an RCS=1m^2 target at reference 3km,
% BEFORE pulse compression and coherent Doppler integration.
% Mixed difficult and easy cases: natural misses should occur.
refRange_m = 3000;
% Reference input SNR selected by difficultyProfile above.
% Keep receiver noise unchanged; the lower difficulty comes from target SNR
% and milder weather parameters, not from disabling complex AWGN.
noiseRMS = 1;

% 2-D CA-CFAR; Pfa is NOMINAL per cell in ideal iid exponential noise.
% Coherent processing, sidelobes and correlated cells invalidate exact Pfa.
% Nominal CFAR Pfa selected by difficultyProfile above.
trainingRange = 8;
trainingDoppler = 4;
guardRange = 3;
guardDoppler = 1;
matchRangeTolerance_m = 22.5;   % 3 range bins
matchVelocityTolerance_mps = 3.538; % 2 Doppler bins
maxDetectionsPerFrame = numRangeBins*numPulses; % No top-3 clipping of false alarms.
sidelobeRange_m = 150;
sidelobeMinGap_dB = 22;

% For incoherent sum of M independent complex-noise receivers, the
% noise-only cell power is Gamma(M), not exponential (Gamma(1)).
% Calculate the matched CA-CFAR nominal threshold multiplier using
% the exact beta/F ratio under independent homogeneous noise cells.
fullNTrain = (2*trainingRange+1)*(2*trainingDoppler+1) - ...
             (2*guardRange+1)*(2*guardDoppler+1);
switch lower(detectionPowerMode)
    case 'array_noncoherent'
        cfarNoiseShape = numRx;
    case 'single_rx'
        cfarNoiseShape = 1;
    otherwise
        error('Unknown detectionPowerMode: %s', detectionPowerMode);
end
cfarAlpha = calculateCfarAlpha(pfaNominal, cfarNoiseShape, fullNTrain);
fprintf('Detector=%s | noise channels=%d | nominal CFAR alpha=%.4f\n', ...
    detectionPowerMode,cfarNoiseShape,cfarAlpha);

[kRain,alphaRain] = rainCoeffsITU838H(fc/1e9);
fprintf('fc=%.1f GHz, B=%.1f MHz, range resolution=%.2f m\n', ...
    fc/1e9,B/1e6,c/(2*B));
fprintf('PRI=%.3f us, pulse count=%d, dv=%.3f m/s, vmax=%.2f m/s\n',...
    PRI*1e6,numPulses,velStep,lambda/(4*PRI));
fprintf('ITU-R P.838-3 horizontal rain coefficients at %.1fGHz: k=%.6f alpha=%.5f\n',...
    fc/1e9,kRain,alphaRain);
fprintf('8Rx ULA: spacing=%.3f mm, angle grid=[-60,60] deg / 0.5 deg\n', ...
    rxSpacing_m*1e3);

%% 2. Pre-create target scenes (group by scene for train/val/test)
scenes = repmat(struct('R',[],'v',[],'az',[],'RCS',[], ...
    'snrRef_dB',[],'snrLevel_dB',[],'split',''),numScenes,1);
for s=1:numScenes
    % IMPORTANT: Separate target-free allocation from the SNR cycle.
    % All five SNR levels occur once in every consecutive 5 scenes.
    % Make one out of every five such *cycles* target-free, so EACH
    % SNR bin has target-present and target-absent examples.
    snrCycle = floor((s-1)/numel(referenceSNRLevels_dB));
    if mod(snrCycle,5)==4
        nTargets=0;                    % 20% target-free at EACH SNR
    else
        nTargets=randi([1 3]);
    end
    if nTargets>0
        R=sort(sceneRangeMin_m + ...
            (sceneRangeMax_m-sceneRangeMin_m)*rand(1,nTargets));
        while any(diff(R)<200)
            R=sort(sceneRangeMin_m + ...
                (sceneRangeMax_m-sceneRangeMin_m)*rand(1,nTargets));
        end
        v=-44+88*rand(1,nTargets);
        % Angles are independently sampled. These are SCENE TRUTH
        % used to synthesize spatial phase, never read by DBF estimator.
        az=-55+110*rand(1,nTargets);
        if strcmpi(difficultyProfile,'demo_easy')
            RCS=rcsMin_m2+(rcsMax_m2-rcsMin_m2)*rand(1,nTargets);
        else
            RCS=10.^(-0.7+1.2*rand(1,nTargets)); % preserve prior distribution
        end
    else
        R=[]; v=[]; az=[]; RCS=[];
    end
    snrLevel = referenceSNRLevels_dB(mod(s-1,numel(referenceSNRLevels_dB))+1);
    scenes(s).R = R;
    scenes(s).v = v;
    scenes(s).az = az;
    scenes(s).RCS = RCS;
    scenes(s).snrLevel_dB = snrLevel;
    scenes(s).snrRef_dB = snrLevel + (2*rand-1);
    if s<=floor(.70*numScenes)
        scenes(s).split='train';
    elseif s<=floor(.85*numScenes)
        scenes(s).split='val';
    else
        scenes(s).split='test';
    end
end

%% 3. Allocate tables
nFrames = numScenes*numel(weatherList);
frameRows = cell(nFrames,22);
targetRows = cell(nFrames*3,15);
targetRowsCount = 0;
example = struct();
frameId=0;

%% 4. Generate each scene for clear / rain / fog
for s=1:numScenes
    scene=scenes(s);
    nTargets=numel(scene.R);
    for w=1:numel(weatherList)
        frameId=frameId+1;
        weather=weatherList{w};

        % Same receiver noise for 3 weather variants of the same scene.
        rng(100000+s,'twister');
        % Independent complex receiver noise for each coherent array element.
        rxCube = noiseRMS/sqrt(2) * complex( ...
            randn(Nfast,numPulses,numRx,'single'), ...
            randn(Nfast,numPulses,numRx,'single'));

        % Environment coefficient in dB/km and simplified scatterers
        rainRate=0;
        fogLWC=0;
        temperatureC=15;
        gammaWeather=0;
        nWeatherScatterers=0;
        if strcmp(weather,'rain')
            rainRate=rainRateMin_mmph + ...
                (rainRateMax_mmph-rainRateMin_mmph)*rand();
            gammaWeather=kRain*rainRate^alphaRain;
            nWeatherScatterers=rainScatterers;
        elseif strcmp(weather,'fog')
            fogLWC=fogLWCMin_gm3 + ...
                (fogLWCMax_gm3-fogLWCMin_gm3)*rand(); % g/m^3
            temperatureC=5+15*rand();
            Kl=fogLiquidCoefficient(fc/1e9,temperatureC);
            gammaWeather=Kl*fogLWC;
        end

        % Every target contributes a time-delayed chirp for each pulse.
        % Slow-time phase exp(j*2*pi*fd*n*PRI) encodes velocity.
        targetSNRs_dB = zeros(1,nTargets);
        twoWayLosses_dB = zeros(1,nTargets);
        for k=1:nTargets
            Rt=scene.R(k);
            vk=scene.v(k);
            fd=2*vk/lambda;
            tau=2*Rt/c;
            td=tFast-tau;
            idx=(td>=0 & td<Tp);
            loss_dB=2*gammaWeather*(Rt/1000);
            twoWayLosses_dB(k)=loss_dB;
            % 1m2 / 3km reference target input SNR, before integration.
            A=10^(scene.snrRef_dB/20)*sqrt(scene.RCS(k)) ...
                *(refRange_m/Rt)^2*10^(-loss_dB/20);
            targetSNRs_dB(k)=20*log10(max(A,realmin));
            echoFast=single(A*exp(1j*pi*K*(td(idx)-Tp/2).^2) ...
                .*exp(1j*2*pi*fd*td(idx)) ...
                .*exp(-1j*2*pi*fc*tau));
            slowPhase=single(exp(1j*2*pi*fd*PRI*pulseIDs));
            echoBlock = echoFast*slowPhase; % fast-time samples x 64 pulses
            % Far-field spatial delay is approximated by narrowband phase.
            spatial = single(exp(1j*2*pi*(rxSpacing_m/lambda) ...
                * rxIndex*sind(scene.az(k))));
            for a=1:numRx
                rxCube(idx,:,a) = rxCube(idx,:,a) + echoBlock*spatial(a);
            end
        end

        % Rain echoes: randomly distributed, moving point scatterers.
        % This is a phenomenological teaching model; NOT ITU rain clutter.
        for m=1:nWeatherScatterers
            Rc=500+5350*rand();
            vc=-12+24*rand();
            fdc=2*vc/lambda;
            tauc=2*Rc/c;
            td=tFast-tauc;
            idx=(td>=0 & td<Tp);
            scatteringA=(rainClutterAmpMin + ...
                (rainClutterAmpMax-rainClutterAmpMin)*rand()) ...
                *(refRange_m/max(Rc,refRange_m))^2;
            scatteringA=scatteringA*10^(-2*gammaWeather*(Rc/1000)/20);
            phase0=2*pi*rand();
            echoFast=single(scatteringA*exp(1j*pi*K*(td(idx)-Tp/2).^2) ...
                .*exp(1j*2*pi*fdc*td(idx)+1j*phase0));
            slowPhase=single(exp(1j*2*pi*fdc*PRI*pulseIDs));
            rainAz_deg = -60+120*rand(); % Each scatterer has its own angle.
            echoBlock = echoFast*slowPhase;
            spatial = single(exp(1j*2*pi*(rxSpacing_m/lambda) ...
                * rxIndex*sind(rainAz_deg)));
            for a=1:numRx
                rxCube(idx,:,a) = rxCube(idx,:,a) + echoBlock*spatial(a);
            end
        end

        %% 5. Pulse compression (fast-time) and Doppler FFT (slow-time)
        RDcomplex = complex(zeros(numRangeBins,numPulses,numRx,'single'));
        windowDoppler = single(dopplerWindow);
        % Processing one receiver at a time avoids an 8-channel FFT temp cube.
        for a=1:numRx
            compressedA = ifft(fft(rxCube(:,:,a),fftLen,1).*H,[],1);
            rangePulsesA = compressedA(rangeSampleIndex,:); % 800 x 64
            RDcomplex(:,:,a) = fftshift( ...
                fft(rangePulsesA.*windowDoppler,[],2),2);
        end
        clear rxCube compressedA rangePulsesA
        if enableZeroDopplerNotch
            RDcomplex(:,numPulses/2+1,:) = 0;
        end
        % Detector power and CFAR alpha must use MATCHED noise statistics.
        % Noncoherent power sum is robust to unknown target azimuth and
        % exploits 8 receiver channels without using ground-truth angles.
        if strcmpi(detectionPowerMode,'array_noncoherent')
            rdPower = sum(double(abs(RDcomplex).^2),3);
        else
            rdPower = double(abs(RDcomplex(:,:,1)).^2);
        end

        % RD intensity for image/save/display, NOT used as angle truth.
        rdMagnitude=single(sqrt(rdPower));
        rdMagnitude=rdMagnitude/max(max(rdMagnitude(:)),eps('single'));
        rdImage_dB=20*log10(max(double(rdMagnitude),1e-6));

        %% 6. 2D CA-CFAR and nonmaximum suppression
        [detRBins,detDBins]=caCfar2d(rdPower,cfarAlpha, ...
            trainingRange,trainingDoppler,guardRange,guardDoppler, ...
            maxDetectionsPerFrame);
        [detRBins,detDBins] = suppressRangeSidelobes(detRBins,detDBins, ...
            rdImage_dB,rangeStep,sidelobeRange_m,sidelobeMinGap_dB);
        detRanges_m = double(rangeAxis(detRBins));
        detVel_mps = double(velAxis(detDBins));

        %% 6.1. Estimate azimuth from 8-channel complex RD snapshots
        % No access to scene.az here. DBF peak is the angle estimate.
        nDet = numel(detRBins);
        detAz_deg = zeros(1,nDet);
        for kk=1:nDet
            arraySnapshot = double(reshape( ...
                RDcomplex(detRBins(kk),detDBins(kk),:),numRx,1));
            beamPower = abs(steeringMatrix' * arraySnapshot).^2;
            [~,bestAngle] = max(beamPower);
            detAz_deg(kk) = azimuthSearch_deg(bestAngle);
        end
        detX_m = double(detRanges_m(:)).*cosd(detAz_deg(:));
        detY_m = double(detRanges_m(:)).*sind(detAz_deg(:));
        clear RDcomplex

        %% 6.2. Upload *predicted* CFAR + DBF targets to HarmonyOS dashboard
        % No ground-truth labels or CNN scores are used for this message.
        % The HTTP server relays the latest frame to the HarmonyOS application.
        if enableDashboardUpload && ~dashboardUploadStopped && ...
                mod(frameId,uploadEveryNFrames)==0
            targetCount = numel(detRBins);
            targetsForApp = cell(1,targetCount); % [] for target-free frames
            for kk=1:targetCount
                rr=detRBins(kk);
                dd=detDBins(kk);
                targetsForApp{kk} = struct( ...
                    'range_m', double(rangeAxis(rr)), ...
                    'velocity_mps', double(velAxis(dd)), ...
                    'azimuth_deg', double(detAz_deg(kk)), ...
                    'x_m', double(detX_m(kk)), ...
                    'y_m', double(detY_m(kk)), ...
                    'peak_db', double(rdImage_dB(rr,dd)));
            end
            dashboardPayload = struct( ...
                'frame_id', double(frameId), ...
                'weather', weather, ...
                'angle_method', 'DBF-8Rx', ...
                'targets', {targetsForApp});
            try
                uploadOptions=weboptions('MediaType','application/json','Timeout',3);
                webwrite(dashboardServerURL,dashboardPayload,uploadOptions);
                fprintf('[Dashboard] Frame %d | weather=%s | detections=%d\n', ...
                    frameId,weather,targetCount);
                if uploadPauseSeconds>0
                    pause(uploadPauseSeconds);
                end
            catch uploadError
                % Simulation and data generation continue even without a server.
                dashboardUploadStopped=true;
                warning(['Dashboard upload failed (uploads disabled until next run). ', ...
                    'Start radar_server.py first. Details: %s'], uploadError.message);
            end
        end

        %% 7. Evaluate with labels ONLY AFTER inference (never as inputs)
        % RD matching: for comparable historic Pd/Precision/F1 and angle RMSE.
        [TP,FP,FN,rangeSSE,velocitySSE,angleSSE,matchedTruth] = matchRDAndAngle( ...
            detRanges_m,detVel_mps,detAz_deg, ...
            scene.R,scene.v,scene.az, ...
            matchRangeTolerance_m,matchVelocityTolerance_mps);
        % Optional stringent 3-D detection: range, velocity AND azimuth pass.
        [TP3D,FP3D,FN3D] = matchRDA3D( ...
            detRanges_m,detVel_mps,detAz_deg, ...
            scene.R,scene.v,scene.az, ...
            matchRangeTolerance_m,matchVelocityTolerance_mps, ...
            matchAzTolerance_deg);

        %% 8. Save image and labels (small files; no raw cubes by default)
        data=struct();
        data.frameID=frameId;
        data.sceneID=s;
        data.split=scene.split;
        data.weather=weather;
        data.rdMagnitude=rdMagnitude;        % 800 x 64 normalized amplitude
        data.rdImage_dB=single(rdImage_dB);  % 800 x 64, max ~0dB
        data.rangeAxis_m=single(rangeAxis);
        data.velocityAxis_mps=single(velAxis);
        data.trueRanges_m=single(scene.R);
        data.trueVelocities_mps=single(scene.v);
        data.trueAzimuth_deg=single(scene.az);  % truth for evaluation ONLY
        data.trueX_m=single(scene.R.*cosd(scene.az));
        data.trueY_m=single(scene.R.*sind(scene.az));
        data.trueRCS_m2=single(scene.RCS);
        data.trueRangeBin1=uint16(round(scene.R/rangeStep)+1); % MATLAB 1-based
        data.trueDopplerBin1=uint8(round((scene.v-velAxis(1))/velStep)+1);
        data.inputSNRRef_dB=scene.snrRef_dB;
        data.targetInputSNR_dB=targetSNRs_dB;
        data.gammaWeather_dBkm=gammaWeather;
        data.rainRate_mmph=rainRate;
        data.fogLWC_gm3=fogLWC;
        data.temperature_C=temperatureC;
        data.numClutterScatterers=nWeatherScatterers;
        data.difficultyProfile=difficultyProfile;
        data.detectedRanges_m=single(detRanges_m);
        data.detectedVelocities_mps=single(detVel_mps);
        data.detectedAzimuth_deg=single(detAz_deg);
        data.detectedX_m=single(detX_m(:).');
        data.detectedY_m=single(detY_m(:).');
        if nDet>0
            data.detectedPeak_dB=single(rdImage_dB( ...
                sub2ind(size(rdImage_dB),detRBins,detDBins)));
        else
            data.detectedPeak_dB=single([]);
        end
        data.TP=TP; data.FP=FP; data.FN=FN;
        data.TP3D=TP3D; data.FP3D=FP3D; data.FN3D=FN3D;
        data.AngleSSE_deg2=angleSSE;
        data.note=['Normalized RD magnitude is KuRALS-like data representation; ',...
            'not identical sensor physics or verified weather scattering.'];
        filename=sprintf('sample_%05d.mat',frameId);
        save(fullfile(outputDir,filename),'data','-v7');

        frameRows(frameId,:)={frameId,s,scene.split,weather,nTargets, ...
            scene.snrLevel_dB,scene.snrRef_dB,rainRate,fogLWC, ...
            gammaWeather,nWeatherScatterers,TP,FP,FN,rangeSSE,velocitySSE, ...
            numel(detRBins),filename,angleSSE,TP3D,FP3D,FN3D};
        for k=1:nTargets
            targetRowsCount=targetRowsCount+1;
            targetRows(targetRowsCount,:)={frameId,s,scene.split,weather,k, ...
                scene.R(k),scene.v(k),scene.RCS(k),twoWayLosses_dB(k), ...
                targetSNRs_dB(k),sprintf('%s',filename), ...
                scene.az(k),scene.R(k)*cosd(scene.az(k)), ...
                scene.R(k)*sind(scene.az(k)),matchedTruth(k)};
        end
        % Use a correctly detected medium/high-SNR rain scene for the
        % example image, rather than the first (lowest-SNR) scene.
        if strcmp(weather,'rain') && nTargets>0
            takeExample = ~isfield(example,'rdDB');
            if ~takeExample && TP>0
                takeExample = example.TP==0 || ...
                    scene.snrLevel_dB > example.snrLevel_dB;
            end
            if takeExample
                % Display relative to median background for visibility;
                % this display scaling does not affect detection.
                logP=10*log10(rdPower+eps);
                example.rdDB=logP-median(logP(:));
                example.R=scene.R;
                example.v=scene.v;
                example.detR=detRanges_m;
                example.detV=detVel_mps;
                example.detAz=detAz_deg;
                example.trueAz=scene.az;
                example.TP=TP;
                example.snrLevel_dB=scene.snrLevel_dB;
            end
        end
        if strcmpi(difficultyProfile,'demo_easy') || ...
                mod(frameId,30)==0 || frameId==1
            fprintf('Generated %d/%d | %s | truth=%d TP=%d FP=%d FN=%d TP3D=%d\n',...
                frameId,nFrames,weather,nTargets,TP,FP,FN,TP3D);
        end
    end
end

%% 9. Export tables
frameVariableNames={'FrameID','SceneID','Split','Weather','NumTargets', ...
    'SNRLevel_dB','InputSNRRef_dB','RainRate_mmph','FogLWC_gm3', ...
    'Gamma_dBkm','NumClutterScatterers','TP','FP','FN', ...
    'RangeSSE_m2','VelocitySSE_m2ps2','NumDetections','FileName', ...
    'AngleSSE_deg2','TP3D','FP3D','FN3D'};
frameTable=cell2table(frameRows,'VariableNames',frameVariableNames);
writetable(frameTable,fullfile(outputDir,'frame_metadata.csv'));

targetNames={'FrameID','SceneID','Split','Weather','TargetID',...
    'TrueRange_m','TrueVelocity_mps','RCS_m2','TwoWayLoss_dB',...
    'TargetInputSNR_dB','FileName','TrueAzimuth_deg','TrueX_m','TrueY_m','Detected2D'};
targetTable=cell2table(targetRows(1:targetRowsCount,:),...
    'VariableNames',targetNames);
writetable(targetTable,fullfile(outputDir,'target_labels.csv'));

%% 10. Report metrics: per-weather and across controlled input SNR
summaryRows=cell(numel(weatherList),19);
for w=1:numel(weatherList)
    mode=weatherList{w};
    mask=strcmp(frameTable.Weather,mode);
    tp=sum(frameTable.TP(mask));
    fp=sum(frameTable.FP(mask));
    fn=sum(frameTable.FN(mask));
    prec=tp/max(tp+fp,1);
    recall=tp/max(tp+fn,1);
    f1=2*prec*recall/max(prec+recall,eps);
    rmseR=NaN; rmseV=NaN;
    if tp>0
        rmseR=sqrt(sum(frameTable.RangeSSE_m2(mask))/tp);
        rmseV=sqrt(sum(frameTable.VelocitySSE_m2ps2(mask))/tp);
    end
    falseFrames=mask & frameTable.NumTargets==0;
    if any(falseFrames)
        pfaFrame=mean(frameTable.FP(falseFrames)>0);
    else
        pfaFrame=NaN;
    end
    if tp>0
        rmseAz=sqrt(sum(frameTable.AngleSSE_deg2(mask))/tp);
    else
        rmseAz=NaN;
    end
    tp3=sum(frameTable.TP3D(mask));
    fp3=sum(frameTable.FP3D(mask));
    fn3=sum(frameTable.FN3D(mask));
    pd3=tp3/max(tp3+fn3,1);
    precision3=tp3/max(tp3+fp3,1);
    f13=2*pd3*precision3/max(pd3+precision3,eps);
    summaryRows(w,:)={mode,sum(mask),tp,fp,fn,recall,prec,f1,...
        fp/sum(mask),pfaFrame,rmseR,rmseV,rmseAz,tp3,fp3,fn3,pd3,f13,sum(falseFrames)};
    fprintf('%5s: Pd=%.3f, Precision=%.3f, F1=%.3f, FA/frame=%.3f, ',...
        mode,recall,prec,f1,fp/sum(mask));
    fprintf('RMSErange=%.2fm, RMSEvel=%.2fm/s, PfaFrame=%.3f\n',...
        rmseR,rmseV,pfaFrame);
end
summaryNames={'Weather','NumFrames','TP','FP','FN','Pd','Precision','F1',...
    'FalseAlarmsPerFrame','PfaTargetFreeFrame','RangeRMSE_m','VelocityRMSE_mps', ...
    'AngleRMSE_deg','TP3D','FP3D','FN3D','Pd3D','F1_3D','NumTargetFreeFrames'};
summaryTable=cell2table(summaryRows,'VariableNames',summaryNames);
writetable(summaryTable,fullfile(outputDir,'evaluation_by_weather.csv'));

% Match the same truth targets and scene IDs across weather for analysis only.
% Count both lost and gained detections; never force an ordering by weather.
ref=targetTable(strcmp(targetTable.Weather,'clear'),:);
refFrames=frameTable(strcmp(frameTable.Weather,'clear'),:);
pairedRows=cell(numel(weatherList),9);
for w=1:numel(weatherList)
    mode=weatherList{w};
    tw=targetTable(strcmp(targetTable.Weather,mode),:);
    fw=frameTable(strcmp(frameTable.Weather,mode),:);
    assert(isequal(tw.SceneID,ref.SceneID) && isequal(tw.TargetID,ref.TargetID), ...
        'Target pairing across weather is inconsistent.');
    assert(isequal(fw.SceneID,refFrames.SceneID), ...
        'Scene pairing across weather is inconsistent.');
    delta=double(fw.FP)-double(refFrames.FP);
    pairedRows(w,:)={mode,height(tw),mean(tw.TwoWayLoss_dB), ...
        median(tw.TwoWayLoss_dB),max(tw.TwoWayLoss_dB), ...
        sum(ref.Detected2D & ~tw.Detected2D), ...
        sum(~ref.Detected2D & tw.Detected2D),mean(delta),sum(delta~=0)};
end
pairedTable=cell2table(pairedRows,'VariableNames', ...
    {'Weather','NumPairedTargets','MeanTwoWayLoss_dB','MedianTwoWayLoss_dB', ...
    'MaxTwoWayLoss_dB','LostDetectionsVsClear','GainedDetectionsVsClear', ...
    'DeltaFPPerFrameVsClear','FramesWithChangedFPCount'});
writetable(pairedTable,fullfile(outputDir,'evaluation_weather_effects.csv'));
disp(pairedTable);

snrRows=cell(numel(referenceSNRLevels_dB)*numel(weatherList),8);
rr=0;
for w=1:numel(weatherList)
    for j=1:numel(referenceSNRLevels_dB)
        rr=rr+1;
        mode=weatherList{w};
        db=referenceSNRLevels_dB(j);
        mask=strcmp(frameTable.Weather,mode) & frameTable.SNRLevel_dB==db;
        tp=sum(frameTable.TP(mask));
        fn=sum(frameTable.FN(mask));
        fp=sum(frameTable.FP(mask));
        % Pd is undefined for a bin with zero real targets.
        if tp+fn>0
            pd=tp/(tp+fn);
        else
            pd=NaN;
        end
        snrRows(rr,:)={mode,db,sum(mask),tp+fn,tp,fn,fp,pd};
    end
end
snrTable=cell2table(snrRows,'VariableNames',...
    {'Weather','InputSNRLevel_dB','NumFrames','NumTrueTargets','TP','FN','FP','Pd'});
writetable(snrTable,fullfile(outputDir,'evaluation_by_snr.csv'));

fprintf('\n===== Per-SNR target-count check =====\n');
disp(snrTable(:,{'Weather','InputSNRLevel_dB','NumFrames','NumTrueTargets','Pd'}));
if any(snrTable.NumTrueTargets==0)
    warning('Some SNR groups have zero true targets: Pd is NaN. Increase numScenes.');
end

%% 11. Figures: a real generated RD frame and Pd-versus-input-SNR
if isfield(example,'rdDB')
    fig1=figure('Name','Synthetic Range-Doppler','Color','w');
    imagesc(velAxis,rangeAxis/1000,example.rdDB);
    axis xy; colormap(parula); colorbar; caxis([-15 25]);
    xlabel('Radial velocity (m/s)'); ylabel('Range (km)');
    cb=colorbar; ylabel(cb,'Power above median background (dB)');
    title(sprintf('Rain RD map, input SNR %.0f dB: circles=truth, crosses=CFAR',example.snrLevel_dB));
    hold on;
    plot(example.v,example.R/1000,'wo','LineWidth',1.6,'MarkerSize',8);
    plot(example.detV,example.detR/1000,'rx','LineWidth',1.1,'MarkerSize',7);
    saveas(fig1,fullfile(outputDir,'example_RD.png'));
    % 2-D ground-plane display. Predicted points use DBF, not truth angle.
    figXY=figure('Name','RDA spatial map','Color','w');
    plot(example.R.*cosd(example.trueAz), ...
         example.R.*sind(example.trueAz),'go','MarkerSize',9, ...
         'LineWidth',1.5); hold on;
    plot(example.detR(:).*cosd(example.detAz(:)), ...
         example.detR(:).*sind(example.detAz(:)), ...
         'rx','MarkerSize',8,'LineWidth',1.5);
    plot(0,0,'ks','MarkerFaceColor','k');
    xlabel('Forward x (m)'); ylabel('Lateral y (m)');
    axis equal; grid on;
    legend('Truth (evaluation only)','CFAR + DBF estimate','Radar', ...
        'Location','best');
    title('Estimated ground-plane positions (x=Rcos(az), y=Rsin(az))');
    saveas(figXY,fullfile(outputDir,'example_XY.png'));
end
fig2=figure('Name','Pd versus input SNR','Color','w');
hold on;
for w=1:numel(weatherList)
    x=referenceSNRLevels_dB;
    y=nan(size(x));
    for j=1:numel(x)
        mask=strcmp(snrTable.Weather,weatherList{w}) & ...
            snrTable.InputSNRLevel_dB==x(j);
        if any(mask), y(j)=snrTable.Pd(find(mask,1)); end
    end
    plot(x,100*y,'-o','LineWidth',1.4,'DisplayName',weatherList{w});
end
xlabel('Nominal reference input SNR (dB)');
ylabel('Target detection probability (%)');
title('2-D CA-CFAR detection versus scene difficulty');
legend('Location','best'); grid on; ylim([0 105]);
saveas(fig2,fullfile(outputDir,'Pd_vs_InputSNR.png'));

configuration=struct('fc',fc,'B',B,'Tp',Tp,'fs',fs,'PRI',PRI, ...
    'numPulses',numPulses,'rangeStep',rangeStep,'velStep',velStep, ...
    'referenceSNRLevels_dB',referenceSNRLevels_dB, ...
    'numScenes',numScenes,'pfaNominal',pfaNominal, ...
    'maxDetectionsPerFrame',maxDetectionsPerFrame, ...
    'sidelobeRange_m',sidelobeRange_m,'sidelobeMinGap_dB',sidelobeMinGap_dB, ...
    'detectionPowerMode',detectionPowerMode, ...
    'cfarNoiseShape',cfarNoiseShape,'cfarAlpha',cfarAlpha, ...
    'enableZeroDopplerNotch',enableZeroDopplerNotch, ...
    'sceneRangeLimits_m',[sceneRangeMin_m sceneRangeMax_m], ...
    'sceneRCSLimits_m2',[rcsMin_m2 rcsMax_m2], ...
    'numRx',numRx,'rxSpacing_m',rxSpacing_m, ...
    'azimuthSearch_deg',azimuthSearch_deg, ...
    'matchAzTolerance_deg',matchAzTolerance_deg, ...
    'difficultyProfile',difficultyProfile, ...
    'rainRateRange_mmph',[rainRateMin_mmph rainRateMax_mmph], ...
    'rainScatterers',rainScatterers, ...
    'rainClutterAmplitudeRange',[rainClutterAmpMin rainClutterAmpMax], ...
    'fogLWCRange_gm3',[fogLWCMin_gm3 fogLWCMax_gm3], ...
    'trainSplit','70%-15%-15% grouped by scene', ...
    'note',['8Rx narrowband ULA; noncoherent array detection and DBF azimuth. ', ...
            'Nominal CA-CFAR theory assumes homogeneous independent noise cells. ', ...
            'demo_easy is a simplified test scenario, not a measured accuracy gain.']);
save(fullfile(outputDir,'configuration.mat'),'configuration');
fprintf('\nDone: %d frames saved in %s\n',nFrames,outputDir);

%% =================== Local functions ===================
function alpha=calculateCfarAlpha(pfa,M,N)
% For P_CUT ~ Gamma(M,scale), and background mean from N independent
% Gamma(M,scale) cells: P_CUT/meanTrain ~ F(2*M,2*M*N).
% Tail probability is betainc(N/(N+alpha), M*N, M).
% Base MATLAB betainc: no Statistics toolbox necessary.
    lo=0;
    hi=100;
    for iter=1:65
        mid=(lo+hi)/2;
        pTail=betainc(N/(N+mid),M*N,M);
        if pTail>pfa
            lo=mid;
        else
            hi=mid;
        end
    end
    alpha=(lo+hi)/2;
end

function [k,alpha]=rainCoeffsITU838H(fGHz)
% ITU-R P.838-3 Annex: exact curve-fit coefficients for horizontal pol.
% Horizontal path (elevation=0), horizontal linear polarization.
    x=log10(fGHz);
    aK=[-5.33980,-0.35351,-0.23789,-0.94158];
    bK=[-0.10008,1.26970,0.86036,0.64552];
    cK=[1.13098,0.45400,0.15354,0.16817];
    aA=[-0.14318,0.29591,0.32177,-5.37610,16.1721];
    bA=[1.82442,0.77564,0.63773,-0.96230,-3.29980];
    cA=[-0.55187,0.19822,0.13164,1.47828,3.43990];
    k=10^(sum(aK.*exp(-((x-bK)./cK).^2)) -0.18961*x +0.71147);
    alpha=sum(aA.*exp(-((x-bA)./cA).^2)) +0.67849*x-1.95537;
end

function Kl=fogLiquidCoefficient(fGHz,tempC)
% ITU-R P.840 dielectric water model for cloud/fog specific attenuation.
    T=tempC+273.15;
    q=300/T-1;
    e0=77.66+103.3*q;
    e1=0.0671*e0;
    e2=3.52;
    fp=20.20-146*q+316*q^2;
    fs=39.8*fp;
    er=(e0-e1)/(1+(fGHz/fp)^2)+...
       (e1-e2)/(1+(fGHz/fs)^2)+e2;
    ei=fGHz*(e0-e1)/(fp*(1+(fGHz/fp)^2))+...
       fGHz*(e1-e2)/(fs*(1+(fGHz/fs)^2));
    eta=(2+er)/ei;
    Kl=0.819*fGHz/(ei*(1+eta^2));
end

function [rows,cols]=caCfar2d(power,alphaCFAR,trainR,trainD,guardR,guardD,maxDetections)
% Vectorized rectangular CA-CFAR in the range-Doppler power plane.
% Small 1-D convolutions implement 2-D rectangular sums without toolboxes.
    P=double(power);
    [nr,nd]=size(P);
    wideSum=conv2(conv2(P,ones(2*trainR+1,1),'same'),...
        ones(1,2*trainD+1),'same');
    guardSum=conv2(conv2(P,ones(2*guardR+1,1),'same'),...
        ones(1,2*guardD+1),'same');
    onesP=ones(size(P));
    wideCount=conv2(conv2(onesP,ones(2*trainR+1,1),'same'),...
        ones(1,2*trainD+1),'same');
    guardCount=conv2(conv2(onesP,ones(2*guardR+1,1),'same'),...
        ones(1,2*guardD+1),'same');
    nTrain=wideCount-guardCount;
    mu=max(wideSum-guardSum,0)./max(nTrain,1);
    % alphaCFAR was calibrated for the actual number of receiver channels.
    % At unmasked interior cells, the full reference window is used.
    mask=P>alphaCFAR.*mu & nTrain>=100;
    % Avoid edges where training window is truncated.
    mask(1:trainR,:)=false;
    mask(nr-trainR+1:end,:)=false;
    mask(:,1:trainD)=false;
    mask(:,nd-trainD+1:end)=false;
    % Local maxima using only built-in functions (no Image Processing Toolbox)
    localMax=true(size(P));
    for rr=-1:1
        for dd=-1:1
            if rr==0 && dd==0, continue; end
            localMax=localMax & (P>circshift(P,[rr,dd]));
        end
    end
    [candidateR,candidateD]=find(mask & localMax);
    if isempty(candidateR)
        rows=zeros(1,0); cols=zeros(1,0); return;
    end
    scores=P(sub2ind(size(P),candidateR,candidateD));
    [~,order]=sort(scores,'descend');
    candidateR=candidateR(order); candidateD=candidateD(order);
    selected=zeros(min(maxDetections,numel(candidateR)),2);
    count=0;
    for ii=1:numel(candidateR)
        r=candidateR(ii); d=candidateD(ii);
        if count>0 && any(abs(selected(1:count,1)-r)<=8 & ...
                          abs(selected(1:count,2)-d)<=4)
            continue;
        end
        count=count+1;
        selected(count,:)=[r,d];
        if count>=maxDetections, break; end
    end
    rows=selected(1:count,1).';
    cols=selected(1:count,2).';
end

function [rows,cols]=suppressRangeSidelobes(rows,cols,rdDB,rangeStep,limit_m,gap_dB)
% CA-CFAR supplies descending-power peaks. No truth is used here.
    if isempty(rows), return; end
    peakDB=rdDB(sub2ind(size(rdDB),rows,cols));
    keep=true(size(rows));
    for j=1:numel(rows)
        if ~keep(j), continue; end
        later=(j+1):numel(rows);
        remove=cols(later)==cols(j) & ...
            abs(rows(later)-rows(j))*rangeStep<=limit_m & ...
            peakDB(j)-peakDB(later)>=gap_dB;
        keep(later(remove))=false;
    end
    rows=rows(keep);
    cols=cols(keep);
end

function [TP,FP,FN,rangeSSE,velSSE,angleSSE,matched] = matchRDAndAngle( ...
    detR,detV,detAz,truthR,truthV,truthAz,tolR,tolV)
% 2-D one-to-one association. Angle error is measured AFTER 2-D matching,
% including very inaccurate angles (no angle gate that biases the RMSE).
    nTruth=numel(truthR);
    nDet=numel(detR);
    matched=false(1,nTruth);
    TP=0; FP=0; rangeSSE=0; velSSE=0; angleSSE=0;
    for j=1:nDet
        if nTruth==0
            FP=FP+1; continue;
        end
        dr=abs(double(truthR)-double(detR(j)));
        dv=abs(double(truthV)-double(detV(j)));
        score=(dr/tolR).^2+(dv/tolV).^2;
        score(matched | dr>tolR | dv>tolV)=inf;
        [best,k]=min(score);
        if isfinite(best)
            matched(k)=true;
            TP=TP+1;
            rangeSSE=rangeSSE+(double(detR(j))-double(truthR(k)))^2;
            velSSE=velSSE+(double(detV(j))-double(truthV(k)))^2;
            angleSSE=angleSSE+(double(detAz(j))-double(truthAz(k)))^2;
        else
            FP=FP+1;
        end
    end
    FN=nTruth-TP;
end

function [TP3,FP3,FN3] = matchRDA3D( ...
    detR,detV,detAz,truthR,truthV,truthAz,tolR,tolV,tolAz)
% One-to-one association, ALL three conditions must be met.
    nTruth=numel(truthR);
    nDet=numel(detR);
    matched=false(1,nTruth);
    TP3=0; FP3=0;
    for j=1:nDet
        if nTruth==0
            FP3=FP3+1; continue;
        end
        dr=abs(double(truthR)-double(detR(j)));
        dv=abs(double(truthV)-double(detV(j)));
        da=abs(double(truthAz)-double(detAz(j)));
        score=(dr/tolR).^2+(dv/tolV).^2+(da/tolAz).^2;
        score(matched | dr>tolR | dv>tolV | da>tolAz)=inf;
        [best,k]=min(score);
        if isfinite(best)
            matched(k)=true;
            TP3=TP3+1;
        else
            FP3=FP3+1;
        end
    end
    FN3=nTruth-TP3;
end
