from spektral.layers import AGNNConv
from tensorflow.keras.layers import LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input, Dense, Reshape
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
from tqdm import tqdm
import museval
import pandas as pd
from tensorflow.keras.layers import Layer
from spektral.utils import normalized_adjacency
from scipy.sparse import csr_matrix
import numpy as np



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
        self.agnn1 = AGNNConv(trainable=True, aggregate='sum', activation=None, name='GraphAttention1')
        self.agnn2 = AGNNConv(trainable=True, aggregate='sum', activation=None, name='GraphAttention2')
    
    def call(self, inputs):
        graph_input, adjacency_input = inputs
        hidden = self.agnn1([graph_input, adjacency_input])
        hidden = LeakyReLU(alpha=0.3)(hidden)
        hidden = self.agnn2([hidden, adjacency_input])
        hidden = LeakyReLU(alpha=0.3)(hidden)
        return hidden
    
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

    # ENCODER ------------------------------------------------------------
    ds4, ds3, ds2, ds1 = encoder(input_shape)
    reshaped = tf.reshape(ds4, shape=(-1, 4, ds4.shape[2]*ds4.shape[3]))

    # ATTENTION GRAPH NEURAL NETWORK -------------------------------------
    num_nodes = 4
    input_dim = ds4.shape[2]*ds4.shape[3]

    # sparse adjacency matrix
    A = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    A_sparse = csr_matrix(A, dtype=np.float32)
    A_coo = A_sparse.tocoo()

    graph_input = tf.keras.Input(shape=(num_nodes, input_dim))
    adjacency_indices = tf.constant(np.column_stack((A_coo.row, A_coo.col)), dtype=tf.int64)
    adjacency_values = tf.constant(A_coo.data, dtype=tf.float32)
    adjacency_shape = tf.constant(A_coo.shape, dtype=tf.int64)

    adjacency_input = tf.SparseTensor(adjacency_indices, adjacency_values, adjacency_shape)

    graph_out = GraphCNN(input_dim)([reshaped, adjacency_input])

    # BACK TO UNET -------------------------------------------------------
    reverted_output = Reshape(target_shape=(4, ds4.shape[2], ds4.shape[3]))(graph_out)
    hadamard_product = tf.keras.layers.multiply([reverted_output, ds4])
    same_dim = hadamard_product

    # BOTTLE NECK --------------------------------------------------------
    bottleneck_output = bottleneck(same_dim)

    # DECODER ------------------------------------------------------------
    decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)

    x = Conv2DTranspose(1, (4, 4), (1, 2), padding="same")(decoder_output)
    output = LeakyReLU(alpha=0.3)(x)

    iwaveunet = Model(input_shape, output, name="AttentionGCN")

    iwaveunet.compile(optimizer='adam', loss='mse')

    history = iwaveunet.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs)

    iwaveunet.save('AttentionGCN.h5')


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
