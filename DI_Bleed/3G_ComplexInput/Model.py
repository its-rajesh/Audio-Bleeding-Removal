#!/usr/bin/env python
# coding: utf-8

# In[77]:


from keras.layers import Input, Dense, ConvLSTM3D,ConvLSTM2D, Conv2DTranspose,\
Flatten, Dropout, BatchNormalization, Reshape, Conv3D, Conv3DTranspose, ConvLSTM1D, Conv1DTranspose
from keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
import os


# In[ ]:


x = Conv3DTranspose(filters = 1, kernel_size = (2, 1, 1), activation='relu')(x)
print(x.shape)


# In[45]:


batch_size = 16
epochs = 100
inp = Input(shape=(2, 3075, 3, 1))


# In[81]:


print(inp.shape)
#x = Conv3D(filters = 1, kernel_size = (2, 1, 1), activation='relu')(inp)

x = ConvLSTM2D(256, (1, 1), activation="tanh", recurrent_activation="hard_sigmoid")(inp)
print(x.shape)

x = ConvLSTM1D(512, 3, activation="tanh", recurrent_activation="hard_sigmoid")(x)
print(x.shape)

x = Conv1DTranspose(256, 3, activation="relu")(x)
print(x.shape)

x = Dense(2 * 3075 * 3 * 256, activation="relu")(x)
x = Reshape((3075, 3, 256))(x)

x = Conv2DTranspose(1, (3, 2), activation="relu")(x)
print(x.shape)


# In[84]:


2 * 3075 * 3 * 256


# In[83]:


1*512


# In[ ]:


##############################################################################################################
#                                                Autoencoder                                                 #
##############################################################################################################

di3_dca = Model(inp, decode)
di3_dca.compile(optimizer="adam", loss="mse")

##############################################################################################################
#                                                TRAINING                                                    #
##############################################################################################################
print("Training...")
cae = di3_dca.fit(
    x=xtrain,
    y=ytrain,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True, verbose=1)

