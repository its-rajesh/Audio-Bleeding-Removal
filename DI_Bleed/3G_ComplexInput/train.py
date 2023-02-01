#!/usr/bin/env python
# coding: utf-8

#  Complex Convolutional Autoencoder Architecture (CCAE) for vocal bleeding removal

# Author: Rajesh R
# Git: its-rajesh
# Date: 23rd December 2022

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, UpSampling3D, Conv2DTranspose,\
Flatten, MaxPooling3D, BatchNormalization, Reshape, Conv3D, Conv3DTranspose, Conv1DTranspose
from tensorflow.keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
import os


path = "/home/anchal/Desktop/rajesh/complex_di/"

print("Loading Dataset")
xtrain = np.load(path+'xtrain.npy')
xtest = np.load(path+'xtest.npy')
ytrain = np.load(path+'ytrain.npy')
ytest = np.load(path+'ytest.npy')
print("Dataset loaded successfully")
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)



batch_size = 16
epochs = 100
inp = Input(shape=(2, 3075, 3, 1))



x = Conv3D(1, (2, 1, 3), activation="relu")(inp)
x = BatchNormalization()(x)
x = MaxPooling3D(pool_size = (1, 3, 1))(x)

x = Conv3D(512, (1, 15, 1), activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling3D(pool_size = (1, 3, 1))(x)

x = Conv3D(1024, (1, 21, 1), activation="relu")(x)
x = BatchNormalization()(x)

x = Conv3DTranspose(512, (1, 21, 1), activation="relu")(x)
x = BatchNormalization()(x)
x = UpSampling3D(size=(1, 3, 1))(x)

x = Conv3DTranspose(1, (2, 15, 3), activation="relu")(x)
x = BatchNormalization()(x)
out = UpSampling3D(size=(1, 3, 1))(x)


complexnn = Model(inp, out)
complexnn.compile(optimizer="adam", loss="mse")

complexnn_train = complexnn.fit(
    x=xtrain,
    y=ytrain,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, verbose=1)
    
    
savemodelpath = './complexnn.h5'
complexnn.save(savemodelpath)
print('FINAL MODEL SAVED SUCCESSFULLY IN ({})'.format(savemodelpath))

print('Testing.....')
loss = complexnn.evaluate(xtest,  ytest, verbose=1)
print("Test Loss: {}".format(loss))

plt.plot(complexnn_train.history['loss'])
plt.legend(['training'], loc = 'upper left')
plt.savefig('loss.png')
print('END')