import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt
from tqdm import tqdm


print('Loading Files...')
path = "/home/anchal/Desktop/rajesh/Datasets/3sourceextensive_bleed/"

outpath = "/home/anchal/Desktop/rajesh/3Source_TDCr/"

Xtrain = np.load(path+'xtrain.npy')
Ytrain = np.load(path+'ytrain.npy')
Xtest = np.load(path+'xtest.npy')
Ytest = np.load(path+'ytest.npy')


X_train = np.vstack([Xtrain, Xtest])
Y_train = np.vstack([Ytrain, Ytest])

print(X_train.shape, Y_train.shape)

print("Calculating A_act")
A_arr = []
for i in tqdm(range(X_train.shape[0])):
    #print(X[i].shape, S[i].shape)
    A_arr.append(X_train[i] @ np.linalg.pinv(Y_train[i]))

A_arr = np.array(A_arr)
print(A_arr.shape)

print("Training..")

n = 3
l = 220500


inp = Input(shape=(n, l, 1))
batch_size = 32
epochs = 600


l1 = Conv2D(16, (1, n), activation='relu')(inp)
l1 = MaxPooling2D(pool_size=(1, 9))(l1)

l2 = Conv2D(32, (1, n), activation='relu')(l1)
l2 = MaxPooling2D(pool_size=(1, 9))(l2)

l3 = Conv2D(64, (1, n), activation='relu')(l2)
l3 = MaxPooling2D(pool_size=(1, 9))(l3)

l4 = Conv2D(128, (1, n), activation='relu')(l3)
l4 = MaxPooling2D(pool_size=(1, 9))(l4)

l5 = Conv2D(256, (1, n), activation='relu')(l4)
l5 = MaxPooling2D(pool_size=(1, 9))(l5)

flatten = Flatten()(l5)

d1 = Dense(256, activation='relu')(flatten)

d2 = Dense(128, activation='relu')(d1)

d3 = Dense(64, activation='relu')(d2)

d4 = Dense(32, activation='relu')(d3)

d5 = Dense(n*n, activation='relu')(d4)

out = Reshape((n, n))(d5)


model = Model(inp, out)

model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
md = model.fit(X_train, A_arr, batch_size=batch_size, epochs=epochs)
model.save(outpath+'NNCr.h5')

plt.plot(md.history['mse'])
plt.savefig(outpath+'epchvserr.png')


print("Calculating A_act for Test")
A_arr = []
for i in tqdm(range(Xtest.shape[0])):
    #print(X[i].shape, S[i].shape)
    A_arr.append(Xtest[i] @ np.linalg.pinv(Ytest[i]))

A_arr = np.array(A_arr)
print(A_arr.shape)
print("Evaluating..")

loss = model.evaluate(Xtest, A_arr)
print(loss)





