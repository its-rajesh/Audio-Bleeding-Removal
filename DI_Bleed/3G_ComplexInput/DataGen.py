####################################################################################################
#                         DRIVER TO GENERATE INPUT DATA (VOCAL: MUSDB18HQ)                         #
####################################################################################################
# Author: Rajesh R (Git: its-rajesh)
# Date: 14/Januray/2023


import os
import librosa as lb
import numpy as np
from tqdm import tqdm

import DynamicInput as dp

path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/test/"
org_path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/"


flag = False
print("Processing Inputs Please Wait...(May take several minutes)")
for i in tqdm(sorted(os.listdir(path))): #INPUT NO OF FILES AT ONCE
    

    bleed, fs = lb.load(path+i+'/vocals.wav', sr=22050)
    true, fs = lb.load(org_path+i+'/vocals.wav', sr=22050)
    
    stft_bleed = lb.stft(bleed, hop_length = 2048)
    stft_true = lb.stft(true, hop_length = 2048)
    
    encode_bleed = dp.DynamicInput.encode3D(stft_bleed, 3, 1)
    encode_true = dp.DynamicInput.encode3D(stft_true, 3, 1)
    
    if not flag:
        xtrain = encode_bleed
        ytrain = encode_true
        flag = True
    else:
        xtrain = np.concatenate((xtrain, encode_bleed), axis=0)
        ytrain = np.concatenate((ytrain, encode_true), axis=0)


print("Writting numpy files...")
np.save('/home/anchal/Desktop/rajesh/complex_di/xtest.npy', xtrain)
np.save('/home/anchal/Desktop/rajesh/complex_di/ytest.npy', ytrain)
