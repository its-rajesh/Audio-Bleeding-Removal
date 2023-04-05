#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import os
import librosa as lb
import soundfile as sf
import museval
from matplotlib import pyplot as plt
from tqdm import tqdm
import librosa.display


# In[ ]:





# In[296]:


name = 'Music Delta - Hendrix/'
seconds = 0

clean_path = "/home/rajesh/Desktop/Datasets/musdb18hq/train/"+name


# In[297]:


vocals, fs = lb.load(clean_path+'vocals.wav')
bass, fs = lb.load(clean_path+'bass.wav')
drums, fs = lb.load(clean_path+'drums.wav')
other, fs = lb.load(clean_path+'other.wav')

mixture, fs = lb.load(clean_path+'mixture.wav')

b1 = int(fs*seconds)
b2 = int(b1+(10*fs))

vocals = vocals[b1:b2]
bass = bass[b1:b2]
drums = drums[b1:b2]
other = other[b1:b2]
mixture = mixture[b1:b2]


# In[309]:


def genbleed(vocals, bass, drums, other, way=1):
    
    if way == 1: # Normal with low bleed
        bvocals = 1 * vocals + 0.02 * bass + 0.09 * drums + 0.016 * other
        bbass = 0.021 * vocals + 1 * bass + 0.15 * drums + 0.16 * other
        bdrums = 0.01 * vocals + 0.12 * bass + 1 * drums + 0.16 * other
        bother = 0.09 * vocals + 0.02 * bass + 0.015 * drums + 1 * other
    if way == 2: # Diagonals varied with low bleed
        bvocals = 0.82 * vocals + 0.02 * bass + 0.09 * drums + 0.016 * other
        bbass = 0.021 * vocals + 0.71 * bass + 0.15 * drums + 0.16 * other
        bdrums = 0.01 * vocals + 0.12 * bass + 0.9 * drums + 0.16 * other
        bother = 0.09 * vocals + 0.02 * bass + 0.015 * drums + 0.66 * other
    if way == 3: # Normal with High Bleed
        bvocals = 1 * vocals + 0.2 * bass + 0.09 * drums + 0.16 * other
        bbass = 0.21 * vocals + 1 * bass + 0.15 * drums + 0.26 * other
        bdrums = 0.1 * vocals + 0.12 * bass + 1 * drums + 0.16 * other
        bother = 0.09 * vocals + 0.2 * bass + 0.15 * drums + 1 * other
    if way == 4: # Ideal high bleed
        bvocals = 1 * vocals + 0.2 * bass + 0.2 * drums + 0.2 * other
        bbass = 0.2 * vocals + 1 * bass + 0.2 * drums + 0.2 * other
        bdrums = 0.2 * vocals + 0.2 * bass + 1 * drums + 0.2 * other
        bother = 0.2 * vocals + 0.2 * bass + 0.2 * drums + 1 * other
    if way == 5: # Ideal low bleed
        bvocals = 1 * vocals + 0.05 * bass + 0.05 * drums + 0.05 * other
        bbass = 0.05 * vocals + 1 * bass + 0.05 * drums + 0.05 * other
        bdrums = 0.05 * vocals + 0.05 * bass + 1 * drums + 0.05 * other
        bother = 0.05 * vocals + 0.05 * bass + 0.05 * drums + 1 * other
    if way == 6: # STANDARD
        bvocals = 1 * vocals + 0.1 * bass + 0.1 * drums + 0.1 * other
        bbass = 0.1 * vocals + 1 * bass + 0.1 * drums + 0.1 * other
        bdrums = 0.1 * vocals + 0.1 * bass + 1 * drums + 0.1 * other
        bother = 0.1 * vocals + 0.1 * bass + 0.1 * drums + 1 * other
        
    return bvocals, bbass, bdrums, bother


# In[310]:


bvocals, bbass, bdrums, bother = genbleed(vocals, bass, drums, other, way=6)


# In[311]:


smix = vocals+bass+drums+other
xmix = bvocals+bbass+bdrums+bother


# In[312]:


np.linalg.norm(mixture-smix)**2, np.linalg.norm(mixture-xmix)**2


# In[313]:


S = np.array([vocals, bass, drums, other])


# In[314]:


s_coeff = mixture @ np.linalg.pinv(S)
s_coeff


# In[315]:


s_coeff = mixture @ np.linalg.pinv(S)
artmix_s_coeff = smix @ np.linalg.pinv(S)
artmix_x_coeff = xmix @ np.linalg.pinv(S)

np.round(s_coeff, 4), np.round(artmix_s_coeff, 4), np.round(artmix_x_coeff, 4)


# In[316]:


outpath = "/home/rajesh/Desktop/"


# In[317]:


sf.write(outpath+"mixture.wav", mixture, fs)
sf.write(outpath+"smix.wav", smix, fs)
sf.write(outpath+"xmix.wav", xmix, fs)


# In[318]:


sf.write(outpath+"bvocals.wav", bvocals, fs)
sf.write(outpath+"bbass.wav", bbass, fs)
sf.write(outpath+"bdrums.wav", bdrums, fs)
sf.write(outpath+"bother.wav", bother, fs)


# In[319]:


sf.write(outpath+"tvocals.wav", vocals, fs)
sf.write(outpath+"tbass.wav", bass, fs)
sf.write(outpath+"tdrums.wav", drums, fs)
sf.write(outpath+"tother.wav", other, fs)


# In[198]:


X = np.array([bvocals, bbass, bdrums, bother])
np.save(outpath+'x5.npy', X)


# In[202]:


x1 = np.load(outpath+"x1.npy")
x2 = np.load(outpath+"x2.npy")
x3 = np.load(outpath+"x3.npy")
x4 = np.load(outpath+"x4.npy")
x5 = np.load(outpath+"x5.npy")


# In[203]:


X_tot = np.array([x1, x2, x3, x4, x5])


# In[204]:


X_tot.shape


# In[ ]:





# # Three Sources Bleed

# In[212]:


name = 'Music Delta - Hendrix/'
seconds = 0

clean_path = "/home/rajesh/Desktop/Datasets/musdb18hq/train/"+name
outpath = "/home/rajesh/Desktop/"


# In[219]:


vocals, fs = lb.load(clean_path+'vocals.wav')
bass, fs = lb.load(clean_path+'bass.wav')
drums, fs = lb.load(clean_path+'drums.wav')

mixture, fs = lb.load(clean_path+'mixture.wav')

b1 = int(fs*seconds)
b2 = int(b1+(10*fs))

vocals = vocals[b1:b2]
bass = bass[b1:b2]
drums = drums[b1:b2]
mixture = mixture[b1:b2]


# In[220]:


def genbleed(vocals, bass, drums, way=1):
    
    if way == 1: # Normal with low bleed
        bvocals = 1 * vocals + 0.02 * bass + 0.09 * drums 
        bbass = 0.021 * vocals + 1 * bass + 0.15 * drums 
        bdrums = 0.01 * vocals + 0.12 * bass + 1 * drums 
    if way == 2: # Diagonals varied with low bleed
        bvocals = 0.82 * vocals + 0.02 * bass + 0.09 * drums 
        bbass = 0.021 * vocals + 0.71 * bass + 0.15 * drums 
        bdrums = 0.01 * vocals + 0.12 * bass + 0.9 * drums 
    if way == 3: # Normal with High Bleed
        bvocals = 1 * vocals + 0.2 * bass + 0.09 * drums 
        bbass = 0.21 * vocals + 1 * bass + 0.15 * drums 
        bdrums = 0.1 * vocals + 0.12 * bass + 1 * drums 
    if way == 4: # Ideal high bleed
        bvocals = 1 * vocals + 0.2 * bass + 0.2 * drums 
        bbass = 0.2 * vocals + 1 * bass + 0.2 * drums 
        bdrums = 0.2 * vocals + 0.2 * bass + 1 * drums 
    if way == 5: # Ideal low bleed
        bvocals = 1 * vocals + 0.05 * bass + 0.05 * drums
        bbass = 0.05 * vocals + 1 * bass + 0.05 * drums 
        bdrums = 0.05 * vocals + 0.05 * bass + 1 * drums 
        
    return bvocals, bbass, bdrums


# In[237]:


bvocals, bbass, bdrums = genbleed(vocals, bass, drums, way=5)


# In[238]:


sf.write(outpath+"bvocals.wav", bvocals, fs)
sf.write(outpath+"bbass.wav", bbass, fs)
sf.write(outpath+"bdrums.wav", bdrums, fs)


# In[239]:


sf.write(outpath+"tvocals.wav", vocals, fs)
sf.write(outpath+"tbass.wav", bass, fs)
sf.write(outpath+"tdrums.wav", drums, fs)


# In[240]:


X = np.array([bvocals, bbass, bdrums])
np.save(outpath+'x5.npy', X)


# In[241]:


x1 = np.load(outpath+"x1.npy")
x2 = np.load(outpath+"x2.npy")
x3 = np.load(outpath+"x3.npy")
x4 = np.load(outpath+"x4.npy")
x5 = np.load(outpath+"x5.npy")


# In[242]:


X_tot = np.array([x1, x2, x3, x4, x5])


# In[245]:


np.save(outpath+'X_tot.npy', X_tot)


# In[246]:


S = np.array([vocals, bass, drums])
S_tot = np.array([S, S, S, S, S])


# In[248]:


np.save(outpath+'S_tot.npy', S_tot)


# In[249]:


def random_gen():
    k = np.round(np.random.uniform(0.001, 0.41), 3)
    return k
def random_diag():
    k = np.round(np.random.uniform(0.6, 1), 3)
    return k

def gen_A():
    A = np.array([[random_diag(), random_gen(), random_gen()],
                  [random_gen(), random_diag(), random_gen()],
                  [random_gen(), random_gen(), random_diag()]])
    return np.round(A, 2)


# In[295]:


gen_A()


# In[ ]:




