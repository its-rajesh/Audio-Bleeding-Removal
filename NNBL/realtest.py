import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt
import librosa as lb
import soundfile as sf
import os

X = np.load("X_tot.npy")
S = np.load("S_tot.npy")

A_act = []
for i in range(5):
    A_act.append(X[i] @ np.linalg.pinv(S[i]))

A_act = np.array(A_act)

model = load_model('NNMF.h5')
predictions = model.predict(X)

loss = model.evaluate(X, A_act)
print(loss)

print(predictions)
print(A_act)