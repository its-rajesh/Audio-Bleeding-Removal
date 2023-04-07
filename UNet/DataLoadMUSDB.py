import numpy as np
import librosa as lb
from tqdm import tqdm


path = "/home/anchal/Desktop/rajesh/Datasets/extensive_bleed/"

print('Loading Raw Data...')
Xtrain = np.load(path+'Xtrain.npy')
Ytrain = np.load(path+'Ytrain.npy')

print('Preparing Training Data...')
Xtrain_stft, Ytrain_stft = [], []
for i in tqdm(range(1500)):
    bleed, true = Xtrain[i], Ytrain[i]

    x1 = lb.stft(bleed[0][:212500], n_fft=2047)
    x2 = lb.stft(bleed[1][:212500], n_fft=2047)
    x3 = lb.stft(bleed[2][:212500], n_fft=2047)
    x4 = lb.stft(bleed[3][:212500], n_fft=2047)
    x = np.array([x1, x2, x3, x4])

    s1 = lb.stft(true[0][:212500], n_fft=2047)
    s2 = lb.stft(true[1][:212500], n_fft=2047)
    s3 = lb.stft(true[2][:212500], n_fft=2047)
    s4 = lb.stft(true[3][:212500], n_fft=2047)
    s = np.array([s1, s2, s3, s4])

    Xtrain_stft.append(x)
    Ytrain_stft.append(s)

print(x.shape)

np.save(path+'Xtrain_stft.npy', Xtrain_stft)
np.save(path+'Ytrain_stft.npy', Ytrain_stft)

print('Preparing Test Data...')
Xtest_stft, Ytest_stft = [], []
for i in tqdm(range(1500, 1588)):
    bleed, true = Xtrain[i], Ytrain[i]

    x1 = lb.stft(bleed[0][:212500], n_fft=2047)
    x2 = lb.stft(bleed[1][:212500], n_fft=2047)
    x3 = lb.stft(bleed[2][:212500], n_fft=2047)
    x4 = lb.stft(bleed[3][:212500], n_fft=2047)
    x = np.array([x1, x2, x3, x4])

    s1 = lb.stft(true[0][:212500], n_fft=2047)
    s2 = lb.stft(true[1][:212500], n_fft=2047)
    s3 = lb.stft(true[2][:212500], n_fft=2047)
    s4 = lb.stft(true[3][:212500], n_fft=2047)
    s = np.array([s1, s2, s3, s4])

    Xtest_stft.append(x)
    Ytest_stft.append(s)

np.save(path+'Xtest_stft.npy', Xtest_stft)
np.save(path+'Ytest_stft.npy', Ytest_stft)

print(s.shape)