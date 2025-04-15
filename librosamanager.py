import librosa

data,sr = librosa.load("data/wavs/bach/wtc1f01.wav")
a = librosa.feature.mfcc(y=data,sr=sr)
data,sr = librosa.load("data/wavs/bach/wtc1f02.wav")
b = librosa.feature.mfcc(y=data,sr=sr)

print()