function [spectrum truespectrum] = processWav( filename )
    filename
    [y, Fs] = wavread(filename);
    full = zeros(1,size(y,1));
    newSize = int32(size(y,1));
    for i = 1:newSize
        full(i) = y(i,1);
    end

    oneChannel = full(1,1:Fs*10);

    ySize = size(oneChannel,2);
    windowSize = 2^12;
    nWindows = int32(ySize/windowSize*2);
    hannWindow = hann(windowSize);
    spectrum = zeros(nWindows,windowSize);
    truespectrum = zeros(size(spectrum));

    for i = 1:nWindows-2
        start = int32((i-1)*windowSize*0.5)+1;
        myWindow = oneChannel(start:(start+windowSize)-1);
        myWindow1 = myWindow'.*hannWindow;
        spectrum(i,:) = abs(fft(myWindow1));
        truespectrum(i,:) = fft(myWindow1);
    end
end

