from spektral.layers import ChebConv
from tensorflow.keras.layers import LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input, Dense, Reshape
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
from tqdm import tqdm
import museval
import pandas as pd

from tensorflow.keras.layers import Layer



def encoder(input_shape):

    #Downsampling block 1
    x = Conv2D(16, (4, 4), padding = "same", dilation_rate=1)(input_shape)
    x = Conv2D(16, (4, 4), padding = "same", dilation_rate=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds1 = Dropout(0.1)(x)

    #Downsampling block 2
    x = Conv2D(32, (4, 4), padding = "same", dilation_rate=2)(ds1)
    x = Conv2D(32, (4, 4), padding = "same", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds2 = Dropout(0.1)(x)

    #Downsampling block 3
    x = Conv2D(64, (4, 4), padding = "same", dilation_rate=4)(ds2)
    x = Conv2D(64, (4, 4), padding = "same", dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds3 = Dropout(0.1)(x)

    #Downsampling block 4
    x = Conv2D(128, (4, 4), padding = "same", dilation_rate=16)(ds3)
    x = Conv2D(128, (4, 4), padding = "same", dilation_rate=16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((1, 2))(x)
    ds4 = Dropout(0.1)(x)

    return ds4, ds3, ds2, ds1

class GraphCNN(Layer):
    def __init__(self, input_dim, **kwargs):
        super(GraphCNN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.gcn1 = ChebConv(64, K=1, activation=None)
        self.gcn2 = ChebConv(input_dim, K=1, activation=None)
        #self.dense = Dense(input_dim, activation='relu')
    
    def call(self, inputs):
        graph_input, adjacency_input = inputs
        hidden = self.gcn1([graph_input, adjacency_input])
        hidden = LeakyReLU(alpha=0.3)(hidden)
        hidden = self.gcn2([hidden, adjacency_input])
        hidden = LeakyReLU(alpha=0.3)(hidden)
        #output = self.dense(hidden)
        return hidden #output
    
    def get_config(self):
        config = super(GraphCNN, self).get_config()
        config['input_dim'] = self.input_dim
        return config


def bottleneck(encoder_output):
    # Bottleneck layer
    x = Conv2D(256, (4, 4), padding = "same")(encoder_output)
    x = Conv2D(256, (4, 4), padding = "same")(x)
    x = BatchNormalization()(x)
    out = LeakyReLU(alpha=0.3)(x)
    
    return out


def decoder(bottleneck_output, ds4, ds3, ds2, ds1):

    #Upsampling Block 4
    x = Conv2DTranspose(128, (4, 4), (1, 1), padding = "same")(bottleneck_output)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds4])

    x = Conv2D(128, (4, 4), padding = "same")(x)
    x = Conv2D(128, (4, 4), padding = "same")(x)
    up4 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 3
    x = Conv2DTranspose(64, (4, 4), (1, 2), padding = "same")(up4)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds3])

    x = Conv2D(64, (4, 4), padding = "same")(x)
    x = Conv2D(64, (4, 4), padding = "same")(x)
    up3 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 2
    x = Conv2DTranspose(32, (4, 4), (1, 2), padding = "same")(up3)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds2])

    x = Conv2D(32, (4, 4), padding = "same")(x)
    x = Conv2D(32, (4, 4), padding = "same")(x)
    up2 = LeakyReLU(alpha=0.3)(x)

    #Upsampling Block 1
    x = Conv2DTranspose(16, (4, 4), (1, 2), padding = "same")(up2)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = concatenate([x, ds1])

    x = Conv2D(32, (4, 4), padding = "same")(x)
    x = Conv2D(32, (4, 4), padding = "same")(x)
    up1 = LeakyReLU(alpha=0.3)(x)

    return up1


def iwaveUnet(Xtrain, Ytrain, batch_size, epochs):

    channels = 4
    samples = 220448
    input_shape = Input(shape=(channels, samples, 1))

    # ENCODER
    ds4, ds3, ds2, ds1 = encoder(input_shape)

    # GRAPH CONVOLUIONAL NEURAL NETWORK
    reshaped = tf.reshape(ds4, shape=(-1, 4, ds4.shape[2]*ds4.shape[3]))
    num_nodes = 4
    input_dim = ds4.shape[2]*ds4.shape[3]
    model = GraphCNN(input_dim)

    A = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    A = np.stack([A]*Xtrain.shape[0])
    #A = tf.convert_to_tensor(A, dtype=tf.float32)

    graph_input = Input(shape=(num_nodes, input_dim))
    adjacency_input = Input(shape=(num_nodes, num_nodes))

    embedding = model([graph_input, adjacency_input])
    x = model([reshaped, A])
    #reverted_output = np.reshape(x, (4, ds4.shape[2], ds4.shape[3]))

    reverted_output = Reshape(target_shape=(4, ds4.shape[2], ds4.shape[3]))(x)
    

    # CONCAT AND REPRESENT OTHER SHAPE
    #con = concatenate([reverted_output, ds4])
    #same_dim = Conv2D(1, (4, 4), padding = "same")(con)

    #hadamard_product = tf.keras.layers.multiply()([reverted_output, ds4])
    hadamard_product = tf.keras.layers.multiply([reverted_output, ds4])
    same_dim = hadamard_product


    # BOTTLE NECK
    bottleneck_output = bottleneck(same_dim)

    # DECODER
    decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)

    x = Conv2DTranspose(1, (4, 4), (1, 2), padding="same")(decoder_output)
    output = LeakyReLU(alpha=0.3)(x)

    iwaveunet = Model(input_shape, output, name="iWaveUnet_ge")

    iwaveunet.compile(optimizer='adam', loss='mse')

    history = iwaveunet.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs)

    iwaveunet.save('iwaveunet_getype3.h5')


    print('Model Saved')



################################################################
#                        DRIVER CODE                           #
################################################################

if __name__ == "__main__":

    dpath = "/home/anchal/Desktop/rajesh/Datasets/extensive_bleed/"

    print('Loading Files...')

    Xtrain = np.load(dpath+'Xtrain.npy')
    Ytrain = np.load(dpath+'Ytrain.npy')
    Xtest = np.load(dpath+'Xtest.npy')
    Ytest = np.load(dpath+'Ytest.npy')


    Xtrain = np.vstack([Xtrain, Xtest])
    Ytrain = np.vstack([Ytrain, Ytest])

    dim = 220448 #Note

    Xtrain = Xtrain[:, :, :dim]
    Ytrain = Ytrain[:, :, :dim]

    print(Xtrain.shape, Ytrain.shape)

    print('Training....')
    batch_size = 1
    epochs = 100
    iwaveUnet(Xtrain, Ytrain, batch_size, epochs)
