import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt
from tqdm import tqdm
import soundfile as sf
import museval
import pandas as pd



print('Loading Files...')
dpath = "/home/anchal/Desktop/rajesh/Datasets/3sourceextensive_bleed/"

path = "/home/anchal/Desktop/rajesh/3Source_TDCr"

X_test = np.load(dpath+'xtest.npy')
Y_test = np.load(dpath+'ytest.npy')

i = 12 #File number
print(X_test.shape)


sf.write(path+'/Outputs/bvocal.wav', X_test[i][0], 22050)
sf.write(path+'/Outputs/bbass.wav', X_test[i][1], 22050)
sf.write(path+'/Outputs/bdrums.wav', X_test[i][2], 22050)

sf.write(path+'/Outputs/tvocal.wav', Y_test[i][0], 22050)
sf.write(path+'/Outputs/tbass.wav', Y_test[i][1], 22050)
sf.write(path+'/Outputs/tdrums.wav', Y_test[i][2], 22050)


print("Evaluating..")

model = tf.keras.models.load_model('NNCr2.h5')

predictions = model.predict(X_test)

fs = 22050

S_pred = np.linalg.inv(predictions[i]) @ X_test[i]
print(predictions[i])

A_act = X_test[i] @ np.linalg.pinv(Y_test[i])
print(A_act)

sf.write(path+'/Outputs/pvocal.wav', S_pred[0], 22050)
sf.write(path+'/Outputs/pbass.wav', S_pred[1], 22050)
sf.write(path+'/Outputs/pdrums.wav', S_pred[2], 22050)


