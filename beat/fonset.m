function [ processed ] = fonset( channel, Fs )

    % STFT

    % size of each window (can play with this if you want)
    windowSize = 2^12;

    nWindows = floor(numel(channel)/windowSize*2);
    spec = zeros(windowSize,nWindows-2);
    for i = 1:nWindows-2
        start = (i-1)*windowSize/2+1;
        window = channel(start:(start+windowSize-1)).*hann(windowSize);
        spec(1:end,i) = abs(fft(window));
    end

    % Derived signals

    % Derivative
    dsdt = spec(1:end,2:end) - spec(1:end,1:(end-1));
    ds2dt2 = dsdt(1:end,2:end) - dsdt(1:end,1:(end-1));

    % Weighted spectrum
    weights = repmat([(1:size(spec,1)/2).^2'; (size(spec,1)/2:-1:1).^2'],1,size(spec,2));
    wspec = spec .* weights;
    dwdt = wspec(1:end,2:end) - wspec(1:end,1:(end-1));
    dw2dt2 = dwdt(1:end,2:end) - dwdt(1:end,1:(end-1));

    % Euclidean distance
    dist = sum((spec(1:end,2:end) - spec(1:end,1:(end-1))).^2);
    
    % Compute onsets (sum)
    
    input = sum(dwdt);

    processed = [];
    lockout = -1;
    for i = 1:size(input,2)
        onset = 0; % guess no onset
        s = sum(input(:,i));

        history = input(:,floor(max(1,i-3*Fs/windowSize)):i);
        if(s > mean(history) + 1.25*std(history))
            onset = 1; % detected energy variation
        end

        if (onset == 1 && lockout < 0)
            processed = [processed i*windowSize/2];

            % don't let onsets pile up on each other
            lockout = floor(Fs/(windowSize*4));
        end

        lockout = lockout - 1;
    end
end

