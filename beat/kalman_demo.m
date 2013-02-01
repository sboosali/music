clear all;

%% Artificial data input (onsets)

% generate on the beat onsets at tempo of 44100
processed = (1:1000).*44100;

% add some gaussian noise
std = 44100*0.05;
for i = 1:size(processed, 2)
    processed(i) = processed(i) + randn()*std;
end

%% Initialize

% parameters (todo: learn these a la Gibbs)
q = 100;    % start with a big q, lock it down after burn-in period   
Rk = std;   % cheat and copy actual

% constants
H = [1 0];
I = eye(2);

% used by Kalman filter
Pk = zeros([2,2]);
xk = zeros([2,1]);

% initialize this (this may not be an optimal initial state)
oldPk = 44100.*ones([2,2])./2;

% first element: assume that first onset is on the beat
% (this assumption's not necessarily correct in general--think pick-ups)
% second element: guess "randomly" that tempo is 35000
% (obviously we know that it's really 44100, but assume we don't know that)
oldxk = [processed(1) 35000]';

%% Kalman filter

results = zeros(numel(processed)-1,5);
for k = 2:numel(processed)
    yk = processed(k); %onset position

    % This is how you solve burn-in (start with big q and make it small later)
    if k == 30
        q = 0.1;
    end

    y = 1; % assume onsets are on the beat
    A = [1 y; 0 1];

    % Kalman Predict
    Qk = q*[y^3/3 y^2/2; y^2/2 y]; % innovation noise covariance
    Pk(1:end, 1:end) = A*oldPk(1:end, 1:end)*A' + Qk;
    Wk = H*Pk(1:end, 1:end)*H'+Rk; % residual (innovation) covariance
    xk(1:end, 1:end) = A*oldxk(1:end, 1:end); % predicted onset/tempo

    % p(yk|y1:k-1,c1:k):
    pyk = normpdf(yk, H*xk, Wk)+eps;

    % [onset #, position, position guess, tempo guess, (un-)certainty]
    results(k-1,1:end) = [k yk/44100 xk(1)/44100 xk(2)/44100 pyk*100000];

    % Kalman Update
    residualError = yk - H*xk(1:end, 1:end);
    Kk = Pk(1:end, 1:end)*H'*(Wk)^(-1); % optimal Kalman gain
    xk(1:end, 1:end) = xk(1:end, 1:end) + Kk*residualError;
    Pk(1:end, 1:end) = (I-Kk*H)*Pk(1:end, 1:end);
    
    % todo: sample instead of maximize? theoretically better?
    oldPk = Pk;
    oldxk = xk;
end

%% Graphs and stuff

% deviation of actual beat from (known) hidden beat state
subplot(5,1,1)
plot(results(1:end,1) - results(1:end,2));

% deviation of predicted beat from actual beat
subplot(5,1,2)
plot(results(1:end,2) - results(1:end,3));

% deviation of predicted beat from hidden beat
subplot(5,1,3)
plot(results(1:end,1) - results(1:end,3));

% deviation of predicted tempo from actual tempo
subplot(5,1,4)
plot(results(1:end,4));

% probability of observed beat given kalman state
subplot(5,1,5)
plot(results(1:end,5));
