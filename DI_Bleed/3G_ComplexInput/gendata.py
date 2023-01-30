import numpy as np
import os
import librosa as lb
from tqdm import tqdm

path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/"

inp = []
for i in tqdm(os.listdir(path)):
    audio, fs = lb.load(path+i+'/vocals.wav', sr=22050)
    audio = audio[:2646000]
    stft = lb.stft(audio)
    inp.append(stft)
    
    
data = []
for i in inp:
    if i.shape == (1025, 5168):
        data.append(i)

        
data = np.array(data)

xtrain = []
for i in tqdm(data):
    mag = np.abs(i)
    phase = np.angle(i)
    xtrain.append(np.stack((mag, phase)))
    
xtrain = np.array(xtrain)
np.save("/home/anchal/Desktop/rajesh/complexNN/xtest.npy", xtrain)
print(xtrain.shape)
