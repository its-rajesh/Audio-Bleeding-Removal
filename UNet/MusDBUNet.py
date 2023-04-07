import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import librosa as lb
from tensorflow.keras.models import Model, save_model



path = "/home/anchal/Desktop/rajesh/Datasets/extensive_bleed/"

Xtrain = np.load(path+'Xtrain_stft.npy')
Ytrain = np.load(path+'Ytrain_stft.npy')
Xtest = np.load(path+'Xtest_stft.npy')
Ytest = np.load(path+'Ytest_stft.npy')

print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)


def double_conv_block(x, n_filters):

    # Conv2D then ReLU activation
    x = layers.Conv3D(n_filters, (1, 3, 3), padding = "same", activation = "relu")(x)
    # Conv2D then ReLU activation
    x = layers.Conv3D(n_filters, (1, 3, 3), padding = "same", activation = "relu")(x)

    return x
   
def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool3D((1, 2, 2))(f)
    p = layers.Dropout(0.3)(p)

    return f, p
   
def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv3DTranspose(n_filters, (1, 3, 3), (1, 2, 2), padding="same")(x)
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
    inputs = layers.Input(shape=(4, 1024, 416, 1))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 16)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 32)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 64)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 128)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 256)
    

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 128)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 64)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 32)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 16)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "softmax")(u9)
    #print(outputs.shape)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model

unet_model = build_unet_model()

unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss="mse",
                   metrics="mse")
                   
epochs = 100
model_history = unet_model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=2)

print('Testing...')
loss = unet_model.evaluate(Xtest, Ytest)
print(loss)

unet_model.save('UNet.h5')


plt.plot(model_history.history['mse'])
plt.savefig('epchvserr.png')