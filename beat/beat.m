% %% Fake data
%processed = (1:15).*44100
%processed = [1 1.25 1.5 1.75 2 2.5 3 4 5 6 7 8 9 10 11 12 13] .* 44100
%processed = [1 2 3 4 5 6 6.25 6.5 6.75 7 8 9 10 11 12].*44100;
%processed = [1 2 3 4 5 6 7 8 9 10].*44100;
processed = [1 2 3 3.5 4 4.5 5 6 6.75 7 8 9 9.5 10 11 12 12.5 13 14 14.75 15 16 16.5 17 18 19 20] .* 22050;
%processed = [1 1.5 2 2.5 3 3.5 4 5 6 7 8 9 10 11 12 13 14 15] .* 22040;
std = 44100*0.01;
for i = 1:size(processed, 2)
    processed(i) = processed(i) + randn()*std;
end

%%

% %% Particle filter parameters 
%tests = [sym(1/4) sym(1/3) sym(1/2) sym(2/3) sym(3/4) sym(1) sym(3/2) sym(2)];
%tests = [sym(1/4) sym(1/2) sym(3/4) sym(1/3) sym(2/3) sym(1) sym(3/2) sym(2) sym(3) sym(4)];
%tests = [sym(1/4) sym(1/2) sym(3/4) sym(1) sym(3/2) sym(2) sym(3) sym(4)];
%tests = [sym(1/2) sym(1) sym(2)];
tests = [sym(1/4) sym(1/2) sym(1) sym(3/2) sym(2) sym(3) sym(4)];
S = numel(tests);
N = 6;
lambda = 1;
%q=0.01
q = 0.1;
% turn down q
Rk = 0.05*44100;

% constants
H = [1 0];
I = eye(2);

% particle consists of: Kalman shit, weights, ckminus1
Pk = 44100.*ones([2,2,N,S])./2;
weights = ones(N, S);
xk = ones([2,1,N,S])./2;
ck = sym(zeros(N,S));

oldPk = 44100.*ones([2,2,N])./2;
finalWeights = ones(1,N);
ckminus1 = sym(zeros(N)); % TODO: "distribute" these?
oldxk = 44100.*ones([2,1,N])./2;
% end particle shit

% tempo prior
for i = 1:N
    oldxk(1:end, 1, i) = [processed(1)/44100 60/((60+(140/(N-1))*(i-1)))].*44100;
end

% %%
R = ones(N*S,1);
C = ones(N*S,1);
beats = [];
tempo = [];
%for k = 2:numel(processed)
for k = 2:200
    k
    if k == 30
        q = 0.001;
    end
    yk = processed(k); %onset positions
    for i = 1:N
      for s = 1:S
        y = sym(tests(s));
        ck(i,s) = sym(sym(ckminus1(i))+y); %score locations
        
        A = [1 y; 0 1];
        % Kalman Predict
        Qk = q*[y^3/3 y^2/2; y^2/2 y]; % innovation noise covariance
        Pk(1:end, 1:end, i,s) = A*oldPk(1:end, 1:end, i)*A' + Qk;
        Wk = H*Pk(1:end, 1:end, i,s)*H'+Rk; % residual (innovation) covariance
        xk(1:end, 1:end, i,s) = A*oldxk(1:end, 1:end, i); % predicted onset/tempo
        
        % p(yk|y1:k-1,c1:k):
        % TODO: should be sqrt(Wk) here?
        pyk = normpdf(yk, H*xk(1:end, 1:end, i, s), Wk)+eps;

        % Kalman Update
        residualError = yk - H*xk(1:end, 1:end, i,s);
        Kk = Pk(1:end, 1:end, i,s)*H'*(Wk)^(-1); % optimal Kalman gain
        xk(1:end, 1:end, i,s) = xk(1:end, 1:end, i,s) + Kk*residualError;
        Pk(1:end, 1:end, i,s) = (I-Kk*H)*Pk(1:end, 1:end, i,s);

        % prior of ck given ck-1
        [n, d] = numden(sym(ck(i,s))-sym(floor(ck(i, s))));
        priorCk = double(exp(-lambda*log2(abs(d))));
        
%        weights(i, s) = finalWeights(i)*pyk*priorCk;
        weights(i, s) = finalWeights(i)*pyk;
      end
    end
    weights = weights/sum(weights(:))
    fprintf('-----\n')
    [t,indices] = sort(weights(:));
    [R,C] = ind2sub(size(weights),indices);
    R = flipud(R);
    C = flipud(C);
    temp = ckminus1(1);
    tempxk = oldxk(1,1,1);
    for i = 1:N
        oldPk(1:end, 1:end, i) = Pk(1:end, 1:end, R(i), C(i));
        finalWeights(i) = weights(R(i),C(i));
        ckminus1(i) = ck(R(i),C(i)); %set ckminus1's of new particles
        oldxk(1:end,1:end,i) = xk(1:end,1:end,R(i),C(i)); % set tempos of new particles 
    end
    if floor(ckminus1(1)) ~= floor(temp)
        tracker = (floor(temp));
        beat = floor(tempxk+(1-(temp-floor(temp)))/(ckminus1(1)-temp)*(oldxk(1,1,1)-tempxk));
        beats = [beats beat];
        tracker = tracker + 1;
        while double(tracker) < double(floor(ckminus1(1)))
            beat = floor(beats(end) + oldxk(2,1,1));
            tracker = tracker + 1;
            beats = [beats beat];
        end
    end
    ckminus1(1)
    %beats = [beats oldxk(1,1,1)/44100];
    tempo = [tempo 60/(oldxk(2,1,1)/44100.0)];
    60/(oldxk(2,1,1)/44100.0)
end
plot(tempo);
%plot(beats);
%%
beats = double(beats);
tempChannel = oneChannel;
temp1 = size(beats,2);
%beatChannel = zeros(1,temp1);
lengthS = 1000/44100;
t1 = 0:1:44100*lengthS;
y1 = 3/4*sin((2*pi/(44100/440))*t1);
y2 = y1*4;
st1 = size(t1,2);
sth = int64(st1/2);

for i = 1:size(beats,2)
    for j = 1:st1
        ind = beats(i)+j-250;
%        beatChannel(ind) = y1(j);
        tempChannel(ind) = tempChannel(ind)*0.25;
        tempChannel(ind) = tempChannel(ind)+y1(j);
    end
end
% %% Write .wav file
wavwrite(tempChannel, Fs, 'ciara.wav');
