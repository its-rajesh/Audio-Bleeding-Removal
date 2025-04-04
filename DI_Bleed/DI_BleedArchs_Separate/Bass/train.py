#!/usr/bin/env python
# coding: utf-8

# # Convolutional Autoencoder Architecture (CAE) for bleeding removal

# Author: Rajesh R
# Git: its-rajesh
# Date: 23rd December 2022

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
Flatten, Dropout, BatchNormalization, Reshape
from tensorflow.keras.models import Model

import numpy as np
import os




path = "/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Bass/"

print("Loading Dataset")
xtrain = np.load(path+'xtrain_bass.npy')
xtest = np.load(path+'xtest_bass.npy')
ytrain = np.load(path+'ytrain_bass.npy')
ytest = np.load(path+'ytest_bass.npy')
print("Dataset loaded successfully")

print("Magnitude Value..")
xtrain = np.array([np.abs(i) for i in xtrain])
xtest = np.array([np.abs(i) for i in xtest])
ytrain = np.array([np.abs(i) for i in ytrain])
ytest = np.array([np.abs(i) for i in ytest])

print('Data Shape Obtained:')
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

print("Model Initiated..")

##############################################################################################################
#                                                ARCHITECTURE                                                #
##############################################################################################################

batch_size = 16
epochs = 100
inp = Input(shape=(3075, 3, 1))

################################################## ENCODER ##################################################

# Conv1 Layer #
x = Conv2D(filters = 32, kernel_size = (2, 2), activation='relu')(inp)
x = BatchNormalization()(x)

# Conv2 Layer #
x = Conv2D(filters = 64, kernel_size = (2, 2), activation='relu')(x)
encode = BatchNormalization()(x)

################################################## MIDDLE ###################################################

x = Flatten()(encode)

middle = Dense(100, activation='relu')(x)

dense = Dense(encode.shape[1] * encode.shape[2] * encode.shape[3], activation='relu')(middle)

reshape = Reshape((encode.shape[1], encode.shape[2], encode.shape[3]))(dense)

################################################## DECODER ##################################################

# TConv2 Layer #
x = Conv2DTranspose(filters = 32, kernel_size = (2, 2), activation='relu')(reshape)
x = BatchNormalization()(x)

# TConv1 Layer #
x = Conv2DTranspose(filters = 1, kernel_size = (2, 2), activation='relu')(x)
decode = BatchNormalization()(x)

##############################################################################################################
#                                                Autoencoder                                                 #
##############################################################################################################

di3_dca = Model(inp, decode)
di3_dca.compile(optimizer="adam", loss="mse")

##############################################################################################################
#                                                TRAINING                                                    #
##############################################################################################################
print("Training...")
cae = di3_dca.fit(
    x=xtrain,
    y=ytrain,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, verbose=1)

##############################################################################################################

savemodelpath = './bass_model.h5'
di3_dca.save(savemodelpath)
print('FINAL MODEL SAVED SUCCESSFULLY IN ({})'.format(savemodelpath))

##############################################################################################################

print("Evaluation...")
loss = di3_dca.evaluate(xtest,  ytest, verbose=1)
print("Test Loss: {}".format(loss))

##############################################################################################################
