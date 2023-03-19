import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape

X = np.load('X_2mins.npy')
S = np.load('S_2mins.npy')

n = 4
l = 2646000

inp = Input(shape=(n, l, 1))
batch_size = 32
epochs = 100

l1 = Conv2D(n*l, (n, 3), activation='relu')(inp)
l2 = Dense(1024, activation='relu')(l1)
l3 = Dense(512, activation='relu')(l2)

encoder = Dense(256, activation = 'relu')(l3)

l3_ = Dense(512, activation = 'relu')(encoder)
l2_ = Dense(1024, activation='relu')(l3_)
l1_ = Conv2DTranspose(1, (n, 3), activation='relu')(l2_)
decoder = l1_

model = Model(inp, decoder)

model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
model.fit(X, S, batch_size=batch_size, epochs=epochs)

model.save('NNMF')
