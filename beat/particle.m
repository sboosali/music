clear all;

%% Load audio/get onsets

% Load audio file

[y, Fs] = wavread('king.wav',44100*30);

% only get first channel (two if stereo)
channel = y(:,1);
processed = fonset(channel, Fs);

%% Initialize

% score positions
tests = [sym(1/4) sym(1/3) sym(1/2) sym(2/3) sym(3/4) sym(1)];
S = numel(tests);

% number of particles
N = 3;

% parameters (todo: learn these a la Gibbs)
q = 10;    % this is the innovation covariance parameter
            % start with a big q, lock it down after burn-in period

Rk = 44100*.05;   % this is the noise covariance parameter
            % cheat and copy actual noise covariance for now
%Rk = 44100*.01;
            
lambda = 1; % particle filter score positions prior parameter
            
% constants
H = [1 0];
I = eye(2);

%% Particle Filter

% used by Kalman filter
Pk = zeros([2,2,N,S]);
xk = zeros([2,1,N,S]); % current Kalman position/momentum
ck = sym(zeros(N,S));  % current score position

% weight of each score position
weights = ones(N,S);

% old weights
oldweights = zeros(N,1);

% initialize this (this may not be an optimal initial state)
oldPk = 44100.*ones([2,2,N])./2;

% first element: assume that first onset is on the beat
% (this assumption's not necessarily correct in general--think pick-ups)
% second element: guess "randomly" that tempo is 35000
% (obviously we know that it's really 44100, but assume we don't know that)
%oldxk = [processed(1) (processed(2)-processed(1))]';
%oldxk = [processed(1) 70000]';
%oldxk = [processed(1) 40000]';

% tempo prior
oldxk = zeros([2,1,N]);
%for i = 1:N
%    oldxk(:, 1, i) = [processed(1) 44100*60/((60+(140/(N-1))*(i-1)))];
%end
oldxk(:,:,1) = [processed(1) 40000]';
oldxk(:,:,2) = [processed(1) 70000]';

xklog = oldxk(:,:,1);

% old score position
%oldck = sym(0);
%oldck = sym(1/2);
oldck = sym(zeros(N,1));
for i = 1:N
    oldck(i) = sym(3/4 - 1/3); % first onset is a pickup
end

% tallied score positions
score = zeros(size(processed));
score(1) = oldck(1);

% Particle filter %

beats = [processed(1) - double(oldck(1))*oldxk(2,:,1)];
for k = 2:numel(processed)
%for k = 2:2
    k
    yk = processed(k); %onset position
    
    % interpolate missing beats
    lastb = beats(end);
    while (lastb + 1.5*oldxk(2) < processed(k))
        beats = [beats (lastb + oldxk(2))];
        lastb = lastb + oldxk(2);
    end
    
    % This is how you solve burn-in (start with big q and make it small later)
    if k == 30
        q = 0.1;
    end
    
    parfor i = 1:N % for each particle
        for s = 1:S % for each possible elapsed interval
            y = tests(s);       % normalized current test

            ykm = round(((yk - oldxk(1))/oldxk(2))/double(y));
            yu = y * sym(ykm);  % un-normalized current test

            ck(i,s) = oldck(i)+y;    % normalized score location given test

            A = [1 yu; 0 1];    % dynamical system rule

            % Kalman Predict
            Qk = q*[y^3/3 y^2/2; y^2/2 y]; % innovation noise covariance
            Pk(:, :,i,s) = A*oldPk(:, :, i)*A' + Qk;
            Wk = H*Pk(:, :,i,s)*H'+Rk; % residual (innovation) covariance
            xk(:, :, i,s) = A*oldxk(:, :, i); % predicted onset/tempo

            % p(yk|y1:k-1,c1:k):
            pyk = normpdf(yk, H*xk(:,:,i,s), Wk)+eps;

            % Kalman Update
            residualError = yk - H*xk(:, :,i,s);
            Kk = Pk(:, :,i,s)*H'*(Wk)^(-1); % optimal Kalman gain
            xk(:, :,i,s) = xk(:, :,i,s) + Kk*residualError;
            Pk(:, :,i,s) = (I-Kk*H)*Pk(:, :,i,s);

            % prior of ck given ck-1
            [n, d] = numden(sym(ck(i,s))-sym(floor(ck(i,s))));
            pck = double(exp(-lambda*log2(abs(d))));

            ck(i,s) = oldck(i) + yu; % denormalize the score location

            weights(i,s) = oldweights(i) + log(pyk) + log(pck);
        end
    end
        
    %[val idx] = max(weights);
    
    % todo: sample instead of maximize? theoretically better?
    %oldPk = Pk(1:end,1:end,idx);
    %oldxk = xk(1:end,1:end,idx);
    %oldck = ck(idx);
    
    %oldweights = weights(idx);
    
    %score(k) = ck(idx)
    %xklog = [xklog [oldxk]];
    
    %
    % pick N best particles
    %
    
    % get indices of N largest weights
    [t,indices] = sort(weights(:));
    [R,C] = ind2sub(size(weights),indices);
    R = flipud(R);
    C = flipud(C);
    
    for i = 1:N
        % todo: sample instead of maximize? theoretically better?
        oldPk(:, :, i) = Pk(:, :, R(i), C(i));
        oldxk(:, :, i) = xk(:, :, R(i), C(i));
        oldck(i) = ck(R(i), C(i));
        oldweights(i) = weights(R(i),C(i));
    end
    
    %
    % pick N best particles spawned from same parent as best particle
    %
    %[~, idx] = max(weights)
    %[~, c] = max(max(weights))
    %R = idx(c)
    
    % get indices of N largest weights associated w/ best particle
    %[~,C] = sort(weights(R,:),'descend')
    %for i = 1:N
    %     oldPk(1:end, 1:end, i) = Pk(1:end, 1:end, R, C(i));
    %     oldxk(1:end, 1:end, i) = xk(1:end,1:end, R, C(i));
    %     oldck(i) = ck(R, C(i));
    %     oldweights(i) = weights(R,C(i));
    %end
    
    score(k) = oldck(1)
    xklog = [xklog oldxk(1:end, 1:end, 1)];
    
    % if onset is assigned to a beat
    % and haven't already assigned an onset to this beat
    if (mod(score(k),1) == 0 && (score(k) - score(k-1) ~= 0))
        beats = [beats yk];
    end
end

%% Graph results

% plot onsets
subplot(4,1,1);
scatter(processed,ones(numel(processed),1))
axis([processed(1),processed(end),0,2]);

% plot score positions
subplot(4,1,2);
abspos = cumsum((score(2:end) - score(1:end-1)).*xklog(2,2:end)) + processed(1);
scatter(abspos,ones(numel(score)-1,1));
axis([processed(1),processed(end),0,2]);

% score versus onset error
subplot(4,1,3);
scatter(abspos, (abspos - processed(2:end))/44100);
axis([processed(1),processed(end),-1,1]);

% beats
subplot(4,1,4);
scatter(beats, ones(numel(beats),1))
axis([beats(1),beats(end),0,2]);

%% Mark beats aurally

output = channel;
mark = 3/4*sin((2*pi/(44100/440))*(0:1:floor(Fs/44.1)));
for i = 1:size(beats,2)
    for j = 1:numel(mark)
        if(beats(i) < 0)
            continue; % might be an implicit beat before start (pick-up)
        end
        ind = round(beats(i))+j-250; % todo: how can beats be non-integral?
        output(ind) = output(ind)*0.25;
        output(ind) = output(ind)+mark(j);
    end
end

%% Write annotated .wav file
wavwrite(output, Fs, 'beats.wav');