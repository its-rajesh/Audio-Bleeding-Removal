import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import librosa as lb
from tensorflow.keras.models import Model, save_model



path = "/home/anchal/Desktop/rajesh/Datasets/2source/"

X_f = np.load(path+'xtrain_0_50.npy')
S_f = np.load(path+'ytrain_0_50.npy')
X_s = np.load(path+'xtrain_50_100.npy')
S_s = np.load(path+'ytrain_50_100.npy')

Xtrain = np.vstack([X_f, X_s])
Ytrain = np.vstack([S_f, S_s])

Xtrain = Xtrain[:, :, :220480]
Ytrain = Ytrain[:, :, :220480]

Xtest = np.load(path+'xtest_0_50.npy')
Ytest = np.load(path+'ytest_0_50.npy')

Xtest = Xtest[:, :, :220480]
Ytest = Ytest[:, :, :220480]

print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)

def double_conv_block(x, n_filters):
    leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.3)
    x = layers.Conv2D(n_filters, (2, 2), padding = "same", activation = leakyrelu)(x)
    x = layers.Conv2D(n_filters, (2, 2), padding = "same", activation = leakyrelu)(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D((1, 2))(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, (2, 2), (1, 2), padding="same")(x)
    #print('UPSAMPLING')
    #print(x.shape)
    # concatenate 
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model():

    # inputs
    inputs = layers.Input(shape=(2, 220480, 1))

    # encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 16)
    f2, p2 = downsample_block(p1, 32)
    f3, p3 = downsample_block(p2, 64)
    f4, p4 = downsample_block(p3, 128)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 256)
    

    # decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 128)
    u7 = upsample_block(u6, f3, 64)
    u8 = upsample_block(u7, f2, 32)
    u9 = upsample_block(u8, f1, 16)

    # outputs
    outputs = layers.Conv2D(1, (1, 1) , padding="same", activation = "linear")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

unet_model = build_unet_model()


unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss="mse")
                   
epochs = 100
model_history = unet_model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=4)

unet_model.save('UNet.h5')

print('Testing...')
loss = unet_model.evaluate(Xtest, Ytest)
print(loss)



plt.plot(model_history.history['mse'])
plt.savefig('epchvserr.png')