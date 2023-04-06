import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt

files = 15000

def random_gen():
    k = np.round(np.random.uniform(0.001, 0.41), 3)
    return k
def random_diag():
    k = np.round(np.random.uniform(0.6, 1), 3)
    return k

def gen_A():
    A = np.array([[random_diag(), random_gen(), random_gen()],
                  [random_gen(), random_diag(), random_gen()],
                  [random_gen(), random_gen(), random_diag()]])
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



inp = Input(shape=(n, l, 1))
batch_size = 16
epochs = 600


l1 = Conv2D(8, (1, n), activation='relu')(inp)
l1 = MaxPooling2D(pool_size=(1, 9))(l1)

l2 = Conv2D(16, (1, n), activation='relu')(l1)
l2 = MaxPooling2D(pool_size=(1, 9))(l2)

l3 = Conv2D(32, (1, n), activation='relu')(l2)
l3 = MaxPooling2D(pool_size=(1, 9))(l3)

l4 = Conv2D(64, (1, n), activation='relu')(l3)
l4 = MaxPooling2D(pool_size=(1, 9))(l4)

l5 = Conv2D(128, (1, n), activation='relu')(l4)
l5 = MaxPooling2D(pool_size=(1, 9))(l5)

flatten = Flatten()(l5)

d1 = Dense(128, activation='relu')(flatten)

d2 = Dense(64, activation='relu')(d1)

d3 = Dense(32, activation='relu')(d2)

d4 = Dense(16, activation='relu')(d3)

d5 = Dense(n*n, activation='relu')(d4)

out = Reshape((n, n))(d5)


model = Model(inp, out)

model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
md = model.fit(X, A_arr, batch_size=batch_size, epochs=epochs)
model.save('NNMF.h5')

plt.plot(md.history['mse'])
plt.savefig('epchvserr.png')