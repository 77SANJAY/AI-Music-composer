%% AI Music Composition Using CNN-LSTM Hybrid Model (MP3 Version)
clear; clc;

%% Parameters
mp3Folder = 'melodies';
mp3Files = dir(fullfile(mp3Folder, '*.mp3'));
fsTarget = 16000; 
sequenceLength = 50;

X = {};
Y = {};

%% Audio Feature Extractor Setup
afe = audioFeatureExtractor( ...
    'SampleRate', fsTarget, ...
    'Window', hamming(1024,'periodic'), ...
    'OverlapLength', 512, ...
    'mfcc', true);

setExtractorParameters(afe, 'mfcc', 'NumCoeffs', 13);  % Updated method

%% Feature Extraction and Sequence Preparation
for i = 1:length(mp3Files)
    [y, fs] = audioread(fullfile(mp3Folder, mp3Files(i).name));
    y = resample(y, fsTarget, fs);  
    if size(y,2) > 1
        y = mean(y, 2);  
    end

    feats = extract(afe, y);  

    for j = 1:(size(feats,1) - sequenceLength)
        X{end+1} = feats(j:j+sequenceLength-1, :)';  
        Y{end+1} = feats(j+sequenceLength, :);       
    end
end

%% Check if data was collected
if isempty(X)
    error('No training data collected. Ensure MP3 files are placed in "melodies" folder.');
end

%% Format Data
XCell = X;
YMat = cell2mat(Y');

%% Define CNN-LSTM Network
inputSize = size(XCell{1}, 1);  
numHiddenUnits = 128;
numOutputFeatures = size(YMat, 2);

layers = [
    sequenceInputLayer(inputSize)
    convolution1dLayer(3, 64, 'Padding','same')
    reluLayer
    lstmLayer(numHiddenUnits, 'OutputMode','last')
    fullyConnectedLayer(numOutputFeatures)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'Verbose', false);

%% Train the network
net = trainNetwork(XCell, YMat, layers, options);

%% Generate new sequence (Random Seed)
numFramesToGenerate = 30;
seedIndex = randi(length(XCell));  % Pick a random seed
seed = XCell{seedIndex};
generatedSeq = seed;

for i = 1:numFramesToGenerate
    inputSeq = generatedSeq(:, end-sequenceLength+1:end);  
    predicted = predict(net, inputSeq);                    
    generatedSeq = [generatedSeq, predicted'];             
end

%% Convert features to audio (enhanced tone generation)
fs = 44100;
melody = [];

for i = 1:size(generatedSeq, 2)
    mfcc_vector = generatedSeq(:, i);
    freq = 440 + mean(mfcc_vector) * 30;  % Use average MFCC for more tone variation
    t = 0:1/fs:0.1;
    tone = sin(2*pi*freq*t).*exp(-4*t);  
    melody = [melody tone];
end

soundsc(melody, fs);
audiowrite('generated_output.wav', melody, fs);

%% Plot waveform and spectrogram
figure;
subplot(2,1,1);
plot((1:length(melody))/fs, melody);
xlabel('Time (s)');
ylabel('Amplitude');
title('Generated Melody - Waveform');
grid on;

subplot(2,1,2);
spectrogram(melody, 256, 200, 512, fs, 'yaxis');
title('Spectrogram of Generated Melody');