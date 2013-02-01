function [ classifier, tclass ] = train_joint( )

files = dir('train/piano/*.wav');
index = find(~[files.isdir]);

classifier = zeros(2*length(index),4096);
tclass = zeros(size(classifier));
IDX = zeros(16,1);
IDXwork = zeros(8,1);

for i = 1:length(index)
    file = files(index(i)).name;
    [spec tspec] = processWav(strcat('train/piano/',file));
    
    % normalize to unit vector
    classifier(i,:) = sum(spec) / sum(sum(spec));
    tclass(i,:) = sum(tspec) / abs(sum(sum(tspec)));
    
    % and keep track of which pitch it is
    IDXwork(i) = str2num(file(2:end-4));
end

%%
[IDXwork perm] = sort(IDXwork);
classifier(1:8,:) = classifier(perm,:);
tclass(1:8,:) = tclass(perm,:);
IDX(1:8) = IDXwork;

%%

files = dir('train/cello/*.wav');
index = find(~[files.isdir]);

for i = 1:length(index)
    file = files(index(i)).name;
    [spec tspec] = processWav(strcat('train/cello/',file));
    
    % normalize to unit vector
    classifier(8+i,:) = sum(spec) / sum(sum(spec));
    tclass(8+i,:) = sum(tspec) / sum(sum(tspec));
    
    % and keep track of which pitch it is
    IDXwork(i) = str2num(file(2:end-4));
end

%%
[IDXwork perm] = sort(IDXwork);
classifier(9:16,:) = classifier(perm+8,:);
tclass(9:16,:) = tclass(perm,:);
IDX(9:16) = IDXwork;
