import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt

files = 10

def random_gen():
    r = np.random.rand(1)
    if r[0] > 0.5:
        k = r[0] - 0.5
    else:
        k = r[0]
    return k

def gen_A():
    A = np.array([[1, random_gen(), random_gen()],
              [random_gen(), 1, random_gen()],
              [random_gen(), random_gen(), 1]])
    return np.round(A, 2)
   
A_arr = []
for i in range(files):
    A_arr.append(gen_A())
A_arr = np.array(A_arr)

n = 3
l = 220500
X = []
S = np.random.random((files, n, l))
for i in range(files):
	X.append(A_arr[i] @ S[i])

X = np.array(X)


model = load_model('NNMF.h5')
predictions = model.predict(X)

loss = model.evaluate(X, A_arr)
print(loss)
print(A_arr)