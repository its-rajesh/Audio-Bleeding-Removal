import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Lambda, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import layers



print('Loading Files...')
path = "/home/anchal/Desktop/rajesh/Datasets/extensive_bleed/"
path2 = "/home/anchal/Desktop/rajesh/TESTING/Test2/"
path3 = "/home/anchal/Desktop/rajesh/TESTING/Test1/"

outpath = "/home/anchal/Desktop/rajesh/GNNs/"

Xtrain = np.load(path+'Xtrain.npy')
Ytrain = np.load(path+'Ytrain.npy')
Xtest = np.load(path+'Xtest.npy')
Ytest = np.load(path+'Ytest.npy')
x1 = np.load(path2+'xevaluate_lh.npy')
y1 = np.load(path2+'yevaluate_lh.npy')
x2 = np.load(path3+'xevaluate.npy')
y2 = np.load(path3+'yevaluate.npy')

X_train = Xtrain
Y_train = Ytrain
#X_train = np.vstack([Xtrain, Xtest, x1, x2])
#Y_train = np.vstack([Ytrain, Ytest, y1, y2])

print(X_train.shape, Y_train.shape)


n = 4
l = 220500
# Define the model architecture
input_shape = (n, l)
X_input = Input(shape=input_shape, name='X_input')

# Define the graph convolutional layer
def graph_conv_layer(X, F_out):
    X = Lambda(lambda x: K.expand_dims(x, axis=2))(X)
    X = layers.Conv2D(filters=F_out, kernel_size=(1,1), strides=(1,1))(X)
    X = Lambda(lambda x: K.squeeze(x, axis=2))(X)
    X = Dropout(0.5)(X)
    return X

# Define the GNN model
H = graph_conv_layer(X_input, 64)
H = graph_conv_layer(H, 32)
S_pred = Dense(l, activation='linear', name='S_pred')(H)
model = Model(inputs=X_input, outputs=S_pred)
model.compile(optimizer='adam', loss='mse')


X_train = Xtrain
S_train = Ytrain
X_test = Xtest
S_test = Ytest

# Train the model
model.fit(X_train, S_train, epochs=100, batch_size=64, validation_data=(X_test, S_test))

# Evaluate the model on the test data
loss = model.evaluate(X_test, S_test)
print("Test loss:", loss)

'''
# Predict the output for some new input data
S_new = np.random.random((1, n, l))
A = np.array([[1, 0.1, 0.2, 0.1],
              [0.05, 1, 0.1, 0.1],
              [0.1, 0.15, 1, 0.2],
              [0.1, 0.1, 0.05, 1]])
X_new = A @ S_new
S_pred = model.predict(X_new)
'''