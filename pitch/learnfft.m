clear all;

%% Train the classifier
windowSize = 2^12;

%[classifier, tclass] = train_joint(windowSize);

classifier = train(windowSize);

%% Load music here!
[y, Fs] = wavread('prelude.wav',44100*15);
full = zeros(1,size(y,1));
newSize = int32(size(y,1));
for i = 1:newSize
    full(i) = y(i,1);
end

% might not want to load whole thing if it's long
%channel = full(1,1:Fs*20);
channel = full(1,:);

% FFT, processing
ySize = size(channel,2);
nWindows = int32(ySize/windowSize*2);
hannWindow = hann(windowSize);
spectrum = zeros(nWindows,windowSize);
truespectrum = zeros(size(spectrum));

for i = 1:nWindows-2
    start = int32((i-1)*windowSize/2)+1;
    myWindow = channel(start:(start+windowSize)-1);
    myWindow1 = myWindow'.*hannWindow;
    truespectrum(i,:) = fft(myWindow1);
    spectrum(i,:) = abs(fft(myWindow1));
end

%% NMF BITCH
[spectra, coeff] = nnmf(spectrum' ,8);%, 'w0', classifier');

%%
firstNote = spectra(:, 8);
in = ifft(firstNote);
in = repmat(in, 10, 1);
player = audioplayer(in, 44100);
play(player);

%% Pseudoinverse solution

M = pinv(classifier)';

spectra = zeros(nWindows,size(classifier,1));
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % pseudoinverse solution
    spectra(i,:) = M*u';
end

%% NMF solution (multiplicative rule)

spectra = zeros(nWindows,size(classifier,1));
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % NMF
    spectra(i,:) = .5*ones(size(classifier,1),1);
    for k = 1:20 % until convergence
        spectra(i,:) = (spectra(i,:)'.*(classifier*(spectrum(i,:)'./(classifier'*spectra(i,:)')))) ...
                       ./(classifier*ones(size(spectrum(i,:)')));
    end
end

%%
plot(1 - gampdf(1:1:500,2,100)/max(gampdf(1:1:500,2,100)));

%%
scale = 1;
mu = 5.5;
sigma = .5;
stickiness = .04; % inverse

%scale = 1;
%mu = 6.5;
%sigma = .5;
%stickiness = .01;

%scale = 250;
%mu = 1;
%sigma = .5;
%stickiness = 5;

x = (1:1:1800) / scale;
plot(1./(x*pi*sigma.*(1 + ((log(x) - mu)/sigma).^2)))

%%
for i = 2:10
    abs(spectra(i,:) - spectra(i-1,:))
end

%%

mu = 0;
sigma = 2.2;
x = 0:.0001:.5;
plot(2*x, - exp((-(log(x./(1-x)) - mu).^2)/(2*sigma^2))./(x.*(1 - x)));
    
%% gradient descent solution (additive)

spectra = ones(nWindows,size(classifier,1));
for i = 1:nWindows
    i

    spectra(i,:) = zeros(size(classifier,1),1);
    eps = 50;
    
    if (i ~= 1)
        % initialize with assumption of stability
        %spectra(i,:) = spectra(i-1,:);
    end

    for k = 1:100 % until convergence
        if (i < 2)
            spectra(i,:) = spectra(i,:) + ...
                eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)')';
        else
            kgam = 2;
            theta = 1;
            
            % gamma distribution does opposite of what we want
            %dpxx1 = theta^(-1) - (kgam-1)*(abs(spectra(i,:) - spectra(i-1,:)) + .01).^(-1);
            
            % log cauchy distribution
            x = (abs(spectra(i,:) - spectra(i-1,:)) + stickiness) / scale;
            dpxx1 = sigma*(mu^2 - 2*mu + sigma^2 - 2*(mu - 1)*log(x) + (log(x)).^2) ...
                ./ (pi*(x.^2).*(mu^2 + sigma^2 - 2*mu*log(x) + (log(x)).^2).^2);
            %dpxx1=0;
            spectra(i,:) = spectra(i,:) + ...
                eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)' - dpxx1')';
        end
        
%        spectra(i,:) = spectra(i,:) + ...
%            eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)' + spectra(i,:)')';

        % constraint enforcement; project back into domain
        for m = 1:size(spectra,2)
            if spectra(i,m) < 0
                spectra(i,m) = 0;
            end
        end

        if eps > 2
            eps = eps - 1;
        else
            eps = .90*eps;
        end
    end
end

%% Graph output
output = fliplr(spectra); % to print right
surf([output zeros(nWindows,1)]); % need extra padding for surf (wtf)

%% Onsets
dsdt = [spectra(1,1:end); subplus(spectra(2:end,1:end) - spectra(1:(end-1),1:end))];
image(255*flipud(dsdt')/max(max(dsdt)));

%%

input = dsdt';

processed = [];
lockout = -1;
for i = 1:size(input,2)
    onset = 0; % guess no onset
    s = sum(input(:,i));

    history = sum(input(:,floor(max(1,i-20*Fs/windowSize)):i));
    if(s > mean(history))% + .5*std(history))
        onset = 1; % detected energy variation
    end

    if (onset == 1 && lockout < 0)
        processed = [processed i];

        % don't let onsets pile up on each other
        lockout = floor(Fs/(windowSize*4));
    end

    lockout = lockout - 1;
end

subplot(2,1,1);
plot(mean(input));
subplot(2,1,2);
scatter(processed,ones(numel(processed),1));

processed = processed*windowSize/2;

% Mark onsets aurally
output = channel;
mark = 3/4*sin((2*pi/(44100/440))*(0:1:floor(Fs/44.1)));
for i = 1:size(processed,2)
    for j = 1:numel(mark)
        ind = processed(i)+j-250;
        output(ind) = output(ind)*0.25;
        output(ind) = output(ind)+mark(j);
    end
end

% Log an annotated .wav file
wavwrite(output, Fs, 'onsets.wav');

%% Generate spectra time lapse
for i = 1:nWindows
    plot(spectra(i,:))
    pause(windowSize/Fs);
end

%% Generate an interpretive output (won't work; boundary issue insurmountable)
newChannel = zeros(size(oneChannel));
lastWindow = ifft(tclass'*spectra(1,:)');
max(lastWindow)
min(lastWindow)
for i = 2:nWindows-1
    start = int32((i-1)*windowSize/2)+1;
    myWindow = ifft(tclass'*spectra(i,:)');
    myWindow1 = lastWindow(windowSize/2+1:end) + myWindow(1:windowSize/2);
    newChannel(start:(start+windowSize/2)-1) = myWindow1;
    lastWindow = myWindow;
end

% mastering
newChannel = newChannel / max(newChannel);

% Write .wav file
wavwrite(newChannel, Fs, 'gen.wav');
