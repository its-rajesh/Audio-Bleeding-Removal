from spektral.layers import GATConv
from tensorflow.keras.layers import LeakyReLU, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Input, Dense, Reshape
from tensorflow.keras.models import Model, load_model, save_model
import tensorflow as tf
from tensorflow.keras.layers import Layer
from spektral.utils import normalized_adjacency
from scipy.sparse import csr_matrix
import numpy as np


import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from spektral.layers import GATConv





class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.channels = channels
        self.gat_layer = GATConv(
            channels=channels,
            attn_heads=1,
            concat_heads=True,
            dropout_rate=0.2,
            activation=None,
        )
        self.leaky_relu = LeakyReLU(alpha=0.3)

    def call(self, inputs, training=True):
        x, a = inputs  # Features and adjacency matrix
        output = self.gat_layer([x, a], training=training)
        output = self.leaky_relu(output)
        return output

    def get_config(self):
        config = {'channels': self.channels}
        base_config = super(GraphAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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



channels = 4
samples = 220448
input_shape = Input(shape=(channels, samples, 1))

# ENCODER ------------------------------------------------------------
ds4, ds3, ds2, ds1 = encoder(input_shape)
#reshaped = tf.reshape(ds4, shape=(-1, 4, ds4.shape[2]*ds4.shape[3]))

# ATTENTION GRAPH NEURAL NETWORK -------------------------------------
num_nodes = ds4.shape[1]*ds4.shape[2]
num_features = ds4.shape[3]

reshaped = tf.reshape(ds4, shape=(-1, ds4.shape[1]*ds4.shape[2], ds4.shape[3]))
A = np.ones((num_nodes, num_nodes))

GATLayer = GraphAttentionLayer(channels=num_features)

A = tf.constant(A, dtype=tf.float32)
print(type(A), type(reshaped))
graph_out = GATLayer([reshaped, A],  training=True)
print(A.shape, reshaped.shape)

# BACK TO UNET -------------------------------------------------------
reverted_output = Reshape(target_shape=(4, ds4.shape[2], ds4.shape[3]))(graph_out)
hadamard_product = tf.keras.layers.multiply([reverted_output, ds4])

# BOTTLE NECK --------------------------------------------------------
bottleneck_output = bottleneck(hadamard_product)

# DECODER ------------------------------------------------------------
decoder_output = decoder(bottleneck_output, ds4, ds3, ds2, ds1)

x = Conv2DTranspose(1, (4, 4), (1, 2), padding="same")(decoder_output)
output = LeakyReLU(alpha=0.3)(x)

iwaveunet = Model(input_shape, output, name="GAT")

iwaveunet.compile(optimizer='adam', loss='mse')

print('Model Compiled')

iwaveunet.load_weights('/scratch/rajeshr.scee.iitmandi/dfUNet/GAT_weights_20to25.h5')

print(iwaveunet.summary())











import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import os



dpath = "/scratch/rajeshr.scee.iitmandi/Dataset/Saraga/4stem/"
outpath = "/scratch/rajeshr.scee.iitmandi/dfUNet/SaragaOutputs"



#i = 22 
X_saraga = np.load(dpath+'saraga_mixtureas4.npy')
print(X_saraga.shape)

f = [14, 69, 111, 105, 144, 218]

print("Evaluating..")
for i in tqdm(f):

    model = tf.keras.models.load_model('dfUNetcm.h5')
    test = [X_saraga[i][0], X_saraga[i][2], X_saraga[i][1], X_saraga[i][3]]

    predictions = model.predict(np.expand_dims(test, axis=0))
    predictions = np.squeeze(predictions)
    print(predictions.shape)

    savepath = outpath+'/Saraga_Outputs/'+str(i)
    os.makedirs(savepath, exist_ok=True)

    sf.write(savepath+'/bvocal.wav', X_saraga[i][0], 22050)
    sf.write(savepath+'/bviolin.wav', X_saraga[i][2], 22050)
    sf.write(savepath+'/bmridangam.wav', X_saraga[i][1], 22050)
    sf.write(savepath+'/bother.wav', X_saraga[i][3], 22050)

    sf.write(savepath+'/pvocal.wav', predictions[0], 22050)
    sf.write(savepath+'/pviolin.wav', predictions[1], 22050)
    sf.write(savepath+'/pmridangam.wav', predictions[2], 22050)
    sf.write(savepath+'/pother.wav', predictions[3], 22050)
    print('DONE')
