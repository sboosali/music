clear all;

%% Artificial data input (onsets)

% generate on the beat onsets
%actual = (1:100);

% generate rhythmically interesting onsets
actual = [1 2 3 4 5 6 6.5 7 8 9 10 10.25 10.5 10.75 11 12 13 13.5 14 15 16 17 18 20 22 24 26];

% generate slow onsets
%actual = (.5:25.5) * 2;

processed = actual .* 44100; % tempo is 44100; i.e. 60bmp

% add some gaussian noise
std = 44100*0.05;
for i = 1:size(processed, 2)
    processed(i) = processed(i) + randn()*std;
end

%% Initialize

% score positions
tests = [sym(1/4) sym(1/2) sym(3/4) sym(1)];
S = numel(tests);

% parameters (todo: learn these a la Gibbs)
q = 100;    % this is the innovation covariance parameter
            % start with a big q, lock it down after burn-in period

Rk = std;   % this is the noise covariance parameter
            % cheat and copy actual noise covariance for now
            
lambda = .5; % particle filter score positions prior parameter
            

% constants
H = [1 0];
I = eye(2);

%% Particle Filter

% used by Kalman filter
Pk = zeros([2,2,S]);
xk = zeros([2,1,S]); % current Kalman position/momentum
ck = sym(zeros(S,1));  % current score position

% weight of each score position
weights = ones(S,1);

% initialize this (this may not be an optimal initial state)
oldPk = 44100.*ones([2,2])./2;

% first element: assume that first onset is on the beat
% (this assumption's not necessarily correct in general--think pick-ups)
% second element: guess "randomly" that tempo is 35000
% (obviously we know that it's really 44100, but assume we don't know that)
oldxk = [processed(1) 38000]';

% old score position
oldck = sym(0);

% tallied score positions
score = zeros(size(processed));

% Particle filter %

for k = 2:numel(processed)
    k
    yk = processed(k); %onset position
    
    % This is how you solve burn-in (start with big q and make it small later)
    if k == 30
        q = 0.1;
    end
    
    for s = 1:S % for each possible elapsed interval
        y = tests(s);       % normalized current test
        
        ykm = round(((yk - oldxk(1))/oldxk(2))/double(y));
        yu = y * sym(ykm);  % un-normalized current test
        
        ck(s) = oldck+y;    % normalized score location given test
        
        A = [1 yu; 0 1];    % dynamical system rule
        
        % Kalman Predict
        Qk = q*[y^3/3 y^2/2; y^2/2 y]; % innovation noise covariance
        Pk(1:end, 1:end,s) = A*oldPk(1:end, 1:end)*A' + Qk;
        Wk = H*Pk(1:end, 1:end,s)*H'+Rk; % residual (innovation) covariance
        xk(1:end, 1:end,s) = A*oldxk(1:end, 1:end); % predicted onset/tempo

        % p(yk|y1:k-1,c1:k):
        pyk = normpdf(yk, H*xk(1:end,1:end,s), Wk)+eps;

        % Kalman Update
        residualError = yk - H*xk(1:end, 1:end,s);
        Kk = Pk(1:end, 1:end,s)*H'*(Wk)^(-1); % optimal Kalman gain
        xk(1:end, 1:end,s) = xk(1:end, 1:end,s) + Kk*residualError;
        Pk(1:end, 1:end,s) = (I-Kk*H)*Pk(1:end, 1:end,s);
        
        % prior of ck given ck-1
        [n, d] = numden(sym(ck(s))-sym(floor(ck(s))));
        pck = double(exp(-lambda*log2(abs(d))));
        
        ck(s) = oldck + yu; % denormalize the score location
        
        weights(s) = pyk*pck;
    end
    
    [val idx] = max(weights);
    
    % todo: sample instead of maximize? theoretically better?
    oldPk = Pk(1:end,1:end,idx);
    oldxk = xk(1:end,1:end,idx);
    oldck = ck(idx);
    
    score(k) = ck(idx)
end

%% Results

% difference (in seconds) from score positions to note onsets
subplot(4,1,1);
scatter(1+score,1 + score - processed/44100);
axis([1,actual(end),-1,1]);

% rhythmic representation of guesses
subplot(4,1,2);
scatter(1+score,ones(numel(score),1))
axis([1,actual(end),0,2]);     

% rhythmic representation of actual
subplot(4,1,3);
scatter(actual,ones(numel(actual),1))
axis([1,actual(end),0,2]);
