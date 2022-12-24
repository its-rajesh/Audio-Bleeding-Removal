#!/usr/bin/env python
# coding: utf-8

# # Deep Convolutional Architecture (DCA) for bleeding removal

# Author: Rajesh R
# Git: its-rajesh
# Date: 23rd December 2022

##############################################################################################################

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
Flatten, Dropout, BatchNormalization, Reshape
from keras.models import Model, load_model

import numpy as np
import os
import librosa as lb
import cmath
import soundfile as sf

##############################################################################################################

print('loading test samples')

path = "/DATA1/rajeshr/project/musdb_bleeded/"
xtest = np.load(path+'test.npy')
xtest, xtest_phase = np.array([np.abs(i) for i in xtest]), np.array([np.angle(i) for i in xtest])

print('Loading Model')
savemodelpath = './bleed_DCA.h5'
deepconv = load_model(savemodelpath)

print('Predicting samples..')
predictions = deepconv.predict(xtest, verbose=1)

##############################################################################################################

print("Creating unbleed samples with trained model")

def convert_complex(mag, phase):
    print(mag.shape, mag, phase.shape, phase)
    m, n = mag.shape
    complexx = []
    for i in range(m):
        for j in range(n):
            complexx.append(cmath.rect(mag[i][j], phase[i][j]))
    return np.array(complexx).reshape(m, n)


def calc_istft(mag, phase):
    signal = []
    for i in range(len(mag)):
        D = convert_complex(mag[i], phase[i])
        signal.append(lb.istft(D))

    return np.array(signal)


##############################################################################################################

print("Working on ISTFT.....")
reconstructed = calc_istft(predictions, xtest_phase)

##############################################################################################################

print("Writting Audio Files!")
out_path = os.getcwd() + '/Outputs'
os.makedirs(out_path, exist_ok=True)
k = 1
n = len(reconstructed)
for audio in reconstructed:
    sf.write(out_path + '/reconstructed{}.wav'.format(k), audio, 8000)
    k += 1
    if k % 10 == 0:
        print("{} files written out of {}".format(k, n))

print("Completed (2/2)")
print("END")

##############################################################################################################
