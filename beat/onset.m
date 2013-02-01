clear all;

%% Load audio file

[y, Fs] = wavread('king.wav',44100*30);

% only get first channel (two if stereo)
channel = y(1:end,1);

%% Time plot
plot(channel);

%% STFT

% size of each window (can play with this if you want)
windowSize = 2^12;

nWindows = floor(numel(channel)/windowSize*2);
spec = zeros(windowSize,nWindows-2);
for i = 1:nWindows-2
    start = (i-1)*windowSize/2+1;
    window = channel(start:(start+windowSize-1)).*hann(windowSize);
    spec(1:end,i) = abs(fft(window));
end

%% Generate spectra time lapse
for i = 1:nWindows
    plot(spec(i,:))
    pause(windowSize/Fs);
end

%% Derived signals

% Derivative
dsdt = spec(1:end,2:end) - spec(1:end,1:(end-1));
ds2dt2 = dsdt(1:end,2:end) - dsdt(1:end,1:(end-1));

% Weighted spectrum
weights = repmat([(1:size(spec,1)/2).^2'; (size(spec,1)/2:-1:1).^2'],1,size(spec,2));
wspec = spec .* weights;
dwdt = wspec(1:end,2:end) - wspec(1:end,1:(end-1));
dw2dt2 = dwdt(1:end,2:end) - dwdt(1:end,1:(end-1));

% Euclidean distance
dist = sum((spec(1:end,2:end) - spec(1:end,1:(end-1))).^2)

%% Stationary spectrum
image(255*spec/max(max(spec)));

%% Stationary derivative
image(255*dsdt/max(max(dsdt)));

%% Stationary accel
image(255*ds2dt2/max(max(ds2dt2)));

%% Stationary weighted spectrum
%image(255*weights/max(max(weights)));
image(255*wspec/max(max(wspec)));

%% Stationary weighted derivative
%image(255*weights/max(max(weights)));
image(255*dwdt/max(max(dwdt)));

%% Stationary weighted accel
%image(255*weights/max(max(weights)));
image(255*dw2dt2/max(max(dw2dt2)));

%% Sum of frequencies
subplot(7,1,1);
plot(sum(spec));
subplot(7,1,2);
plot(max(0,sum(dsdt)));
subplot(7,1,3);
plot(max(0,sum(ds2dt2)));
subplot(7,1,4);
plot(sum(wspec));
subplot(7,1,5);
plot(max(0,sum(dwdt)));
subplot(7,1,6);
plot(max(0,sum(dw2dt2)));
subplot(7,1,7);
plot(max(0,dist));

%% Compute onsets (very poor algorithm)
processed = [];
lockout = -1;
for i = 1:size(dsdt,2)
    i
    onset = 0; % guess no onset
    for bucket = 1:windowSize
        history = dsdt(bucket,floor(max(1,i-Fs/windowSize)):i);
        if(dsdt(bucket,i) > mean(history) + 6*std(history))
            onset = 1; % detected energy variation > 6sigma from mean
        end
    end
    if (onset == 1 && lockout < 0)
        fprintf('onset\n')
        processed = [processed i*windowSize/2];
        
        % don't let onsets pile up on each other
        lockout = floor(Fs/(windowSize*4))
    end
    
    lockout = lockout - 1;
end

%% Compute onsets (sum)
input = sum(dwdt);

processed = [];
lockout = -1;
for i = 1:size(input,2)
    i
    onset = 0; % guess no onset
    s = sum(input(:,i));
    
    history = input(:,floor(max(1,i-3*Fs/windowSize)):i);
    if(s > mean(history) + 1.25*std(history))
        onset = 1; % detected energy variation
    end

    if (onset == 1 && lockout < 0)
        fprintf('onset\n')
        processed = [processed i*windowSize/2];
        
        % don't let onsets pile up on each other
        lockout = floor(Fs/(windowSize*4))
    end
    
    lockout = lockout - 1;
end

%% Plot onsets

scatter(processed,ones(numel(processed),1))

%% Mark onsets aurally

output = channel;
mark = 3/4*sin((2*pi/(44100/440))*(0:1:floor(Fs/44.1)));
for i = 1:size(processed,2)
    for j = 1:numel(mark)
        ind = processed(i)+j-250;
        output(ind) = output(ind)*0.25;
        output(ind) = output(ind)+mark(j);
    end
end

%% Write annotated .wav file
wavwrite(output, Fs, 'onsets.wav');
