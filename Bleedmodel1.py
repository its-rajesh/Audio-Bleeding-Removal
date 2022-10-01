import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, \
    MaxPool2D, BatchNormalization, MaxPooling2D, \
    Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import os


class RemoveBleed:

    def __init__(self, xtrain, ytrain, xtest, ytest):
        print('Model Initiated')
        self.batch_size = 16
        self.epochs = 20
        self.inp = Input(shape=(257, 938, 1))
        self.out_dim = 257 * 938  # Audio file is of 15seconds duration, fs=8000Hz, mono, n_fft=512.
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        # Create our model layers, training and evaluate
        self.modelLayers()
        self.start_train()
        print('Evaluating Test Dataset')
        self.predictions()


    def modelLayers(self):
        self.x = Conv2D(32, 3, activation='relu')(self.inp)
        self.x = BatchNormalization()(self.x)
        self.x = MaxPooling2D(pool_size=(2, 2))(self.x)
        self.x = Dropout(0.30)(self.x)

        self.x = Conv2D(64, 3, activation='relu')(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = MaxPooling2D(pool_size=(2, 2))(self.x)
        self.x = Dropout(0.30)(self.x)

        self.x = Flatten()(self.x)
        self.x = Dense(100, activation='relu')(self.x)
        self.x = Dropout(0.30)(self.x)

        self.x = Dense(self.out_dim, activation='relu')(self.x)
        self.x = Reshape((257, 938))(self.x)


    def start_train(self):
        self.bleed_removal = Model(inputs=self.inp, outputs=self.x)
        self.bleed_removal.compile(loss='MeanSquaredError', optimizer='adam', metrics='accuracy')
        print(self.bleed_removal.summary)

        path_checkpoint = os.getcwd()+"/cp.ckpt"
        directory_checkpoint = os.path.dirname(path_checkpoint)

        callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                      save_weights_only=True,
                                                      verbose=1)

        print('---------------------------------------------------------')
        print('START OF TRAINING')
        bleed_removal_train = self.bleed_removal.fit(self.xtrain, self.ytrain, epochs=self.epochs, batch_size=self.batch_size, callbacks=[callback])
        print('---------------------------------------------------------')
        print('END OF TRAINING')

    def predictions(self):
        self.predictions = self.bleed_removal.predict(self.xtest, verbose=1)




