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
        self.epochs = 10
        self.inp = Input(shape=(257, 7501, 1))
        self.out_dim = 257 * 7501  # Audio file is of 2 minutes duration, fs=8000Hz, mono, n_fft=512.
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        # Create our model layers, training and evaluate
        self.modelLayers()
        self.start_train()
        #self.evaluation()
        print('Predicting Test Data')
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
        self.x = Reshape((257, 7501))(self.x)

    def start_train(self):
        self.bleed_removal = Model(inputs=self.inp, outputs=self.x)
        self.bleed_removal.compile(loss='MeanSquaredError', optimizer='adam', metrics='accuracy')
        print(self.bleed_removal.summary)

        path = os.getcwd() + '/Checkpoints'
        os.makedirs(path, exist_ok=True)

        path_checkpoint = path + "/cp.ckpt"
        # directory_checkpoint = os.path.dirname(path_checkpoint)

        callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                      save_weights_only=True,
                                                      verbose=1)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=path_checkpoint,
            verbose=1,
            save_weights_only=True,
            save_freq=1)  # for every 5 epochs, model will be saved.

        print('---------------------------------------------------------')
        print('START OF TRAINING')
        self.bleed_removal.fit(self.xtrain, self.ytrain, epochs=self.epochs,
                               batch_size=self.batch_size, callbacks=[cp_callback])
        print('---------------------------------------------------------')
        print('END OF TRAINING')


    def evaluation(self):
        self.loss, self.acc = self.bleed_removal.evaluate(self.xtest,  self.ytest, verbose=2)
        print('---------------------------------------------------------')
        print('EVALUATING THE MODEL')
        print('loss:{}, accuracy:{}'.format(self.loss, self.acc))
        print('---------------------------------------------------------')


    def predictions(self):
        self.predictions = self.bleed_removal.predict(self.xtest, verbose=1)
