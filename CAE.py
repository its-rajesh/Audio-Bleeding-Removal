#!/usr/bin/env python
# coding: utf-8

# # Convolutional Autoencoder Architecture (CAE) for bleeding removal

# Author: Rajesh R
# Git: its-rajesh
# Date: 23rd December 2022

##############################################################################################################

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
Flatten, Dropout, BatchNormalization, Reshape
from keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
import os

##############################################################################################################

# ### Data loading

path = "/DATA1/rajeshr/project/musdb_bleeded/"

print("Loading Dataset")
xtrain = np.load(path+'train.npy')
xtest = np.load(path+'test.npy')
ytrain = np.load(path+'gt_train.npy')
ytest = np.load(path+'gt_test.npy')
print("Dataset loaded successfully")


##############################################################################################################


print("Getting Magnitude & Phase")
xtrain = np.array([np.abs(i) for i in xtrain])
xtest, xtest_phase = np.array([np.abs(i) for i in xtest]), np.array([np.angle(i) for i in xtest])
ytrain = np.array([np.abs(i) for i in ytrain])
ytest = np.array([np.abs(i) for i in ytest])
print("Model Initializing....")

##############################################################################################################

# ### Architecture

batch_size = 16
epochs = 50
inp = Input(shape=(1025, 5168, 1))


# Conv1 #
x = Conv2D(filters = 1, kernel_size = (15, 15), activation='relu')(inp)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.30)(x)


# Conv2 #
x = Conv2D(filters = 32, kernel_size = (9, 9), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.30)(x)


# Conv 3 #
x = Conv2D(filters = 64, kernel_size = (7, 7), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.30)(x)


# Conv 4 #
x = Conv2D(filters = 128, kernel_size = (5, 5), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.30)(x)


# Conv 5 #
x = Conv2D(filters = 512, kernel_size = (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
encoded = Dropout(0.30)(x)


flatten = Flatten()(encoded)
middle = Dense(100, activation='relu')(flatten)
dense = Dense(encoded.shape[1] * encoded.shape[2] * encoded.shape[3], activation='relu')(middle)
reshape = Reshape((encoded.shape[1], encoded.shape[2], encoded.shape[3]))(dense)


# TConv 4 #
x = Conv2DTranspose(filters = 128, kernel_size = (3, 3), activation='relu')(reshape)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Dropout(0.30)(x)


# TConv 3 #
x = Conv2DTranspose(filters = 64, kernel_size = (5, 5), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Dropout(0.30)(x)


# TConv 2 #
x = Conv2DTranspose(filters = 32, kernel_size = (7, 7), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Dropout(0.30)(x)


# TConv 1 #
x = Conv2DTranspose(filters = 1, kernel_size = (9, 9), activation='relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Dropout(0.30)(x)


x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dense(inp.shape[1] * inp.shape[2] * inp.shape[3], activation='relu')(x)
decoded = Reshape((inp.shape[1], inp.shape[2], inp.shape[3]))(x)


print("Model Initiated")

##############################################################################################################

# Autoencoder
deepconv = Model(inp, decoded)
deepconv.compile(optimizer="adam", loss="mse")
#deepconv.summary()

##############################################################################################################

print("Training...")
cae = deepconv.fit(
    x=xtrain,
    y=ytrain,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, verbose=1)

##############################################################################################################

savemodelpath = './bleed_CAE.h5'
deepconv.save(savemodelpath)
print('FINAL MODEL SAVED SUCCESSFULLY IN ({})'.format(savemodelpath))

##############################################################################################################

print("Evaluation...")
loss = deepconv.evaluate(xtest,  ytest, verbose=1)
print("Test Loss: {}".format(loss))

##############################################################################################################

print("Prediction...")
predictions = deepconv.predict(xtest, verbose=1)
np.save('./predictions.npy',predictions)
print('Predictions Saved')
print("Completed (1/3)")

predictions = np.squeeze(predictions, axis=3)

##############################################################################################################


def convert_complex(mag, phase):
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
true = calc_istft(ytest, ytest_phase)

##############################################################################################################

print("Calculating Evaluation Metrics")
sdr, isr, sir, sar = museval.evaluate(true, reconstructed, win=44100, hop=44100, mode='v4', padding=True)


def get_metrics(y):
    avg_y = []
    for i in range(len(y)):
        x = y[~np.isnan(y)]
        avg = sum(x)/len(x)
        avg_y.append(avg)
    return avg_y
    
    
avg_sdr = get_metrics(sdr)
avg_isr = get_metrics(isr) #Source to Spatial Distortion Image
avg_sir = get_metrics(sir)
avg_sar = get_metrics(sar)

print('Overall SDR: ', sum(avg_sdr)/len(avg_sdr))
print('Overall ISR: ', sum(avg_isr)/len(avg_isr))
print('Overall SIR: ', sum(avg_sir)/len(avg_sir))
print('Overall SAR: ', sum(avg_sar)/len(avg_sar))
print("Completed (2/3)")

##############################################################################################################

print("Writting Audio Files!")
out_path = os.getcwd() + '/Outputs'
os.makedirs(out_path, exist_ok=True)
k = 1
n = len(reconstructed)
for audio in reconstructed:
    sf.write(out_path + '/reconstructed{}.wav'.format(k), audio, 22050)
    print("{} files written out of {}".format(k, n))
    k += 1

print("Completed (3/3)")

##############################################################################################################

plt.plot(cae.history['loss'])
plt.legend(['training'], loc = 'upper left')
plt.savefig('loss.png')

print("END")

##############################################################################################################
