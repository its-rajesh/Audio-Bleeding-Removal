import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tqdm import tqdm
import soundfile as sf
import museval
import pandas as pd
import os
import librosa as lb



def check_samples(t):
    n = len(t)
    if n != 220500:
        t = np.pad(t, pad_width=(0, 220500-n))
    
    return t


def preprocess(x):
    block = 220500 # 10 seconds, 10*fs
    split = []
    n = len(x)
    for i in range(0, n, block):
        t = x[i:i+block]
        t = check_samples(t)
        split.append(t)

    return np.array(split)



print('Loading Files...')
dpath = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/train/"
cpath = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_cleaned/"
modelpath = "/home/anchal/Desktop/rajesh/NN_IR/OVERALL/"

model = load_model(modelpath+'NNCr4.h5')

folders = sorted(os.listdir(dpath))

print('Preprocessing....')
for i in tqdm(folders):
    src = dpath + i
    dest = cpath + i
    os.makedirs(dest, exist_ok=True)

    vocals, fs = lb.load(src+'/vocals.wav')
    bass, fs = lb.load(src+'/bass.wav')
    drums, fs = lb.load(src+'/drums.wav')
    other, fs = lb.load(src+'/other.wav')

    vocals = preprocess(vocals)
    bass = preprocess(bass)
    drums = preprocess(drums)
    other = preprocess(other)

    n = vocals.shape[0]
    S_pred = []
    for j in range(n):
        X = np.array([[vocals[j], bass[j], drums[j], other[j]]])
        A = model.predict(X)
        A = np.squeeze(A, axis=0)
        X = np.squeeze(X, axis=0)
        S = np.linalg.inv(A) @ X
        S_pred.append(S)

    S_pred = np.hstack(S_pred)
    sf.write(dest+'/vocals.wav', S_pred[0], 22050)
    sf.write(dest+'/bass.wav', S_pred[1], 22050)
    sf.write(dest+'/drums.wav', S_pred[2], 22050)
    sf.write(dest+'/other.wav', S_pred[3], 22050)






















