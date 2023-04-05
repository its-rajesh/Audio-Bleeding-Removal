#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[66]:


def updateA(X, S):
    
    n, l = X.shape
    I = np.eye(n)
    regularizer = 0.00001
    
    XST = X @ S.T
    SST = S @ S.T
    F1 = (XST + regularizer*I)
    F2 = (SST + regularizer*I)
    F2_inv = np.linalg.inv(F2)
    A = F1 @ F2_inv
    return A


# In[67]:


def updateS(X, A, m, O):
    n, l = X.shape
    #O = np.ones((n, 1))
    ATA = A.T @ A
    ATX = A.T @ X
    F1 = (ATA + O @ O.T) + 0.00001*np.eye(n)
    F1_inv = np.linalg.inv(F1)
    F2 = (O @ m + A.T @ X)
    S = F1_inv @ F2
    return S


# In[68]:


def objective(X, A, S, m, O):
    return np.linalg.norm(X-(A@S))**2 + np.linalg.norm(m-np.sum(np.dot(O.T, S), axis=0))**2


# In[76]:


def project(A, gamma1, gamma2):
    n, _ = A.shape
    for i in range(n):
        for j in range(n):
            if i!=j:
                if A[i][j] > gamma2:
                    A[i][j] = gamma2
                if A[i][j] < gamma1:
                    A[i][j] = gamma1
            else:
                #pass
                A[i][j] = 1
    return A


# In[98]:


def Minimize(X, m, O, gamma1, gamma2, tol=1e-3, max_iter=1000):
    n, l = X.shape
    A = np.eye(n)
    S_opt = X.copy()
    
    iters = 0
    while True:
        
        A_opt = updateA(X, S_opt)
        A_opt = project(A_opt, gamma1, gamma2)
        S_opt = updateS(X, A_opt, m, O)
        loss = objective(X, A_opt, S_opt, m, O)
        #print('ITERATION:', iters+1, 'LOSS:', loss, sep=' ')
        if iters>0 and (prev_loss - loss) <= tol:
            break
        prev_loss = loss
        iters +=1
        if iters >= max_iter:
            print('MAXIMUM ITERATION REACHED!!!')
            break
        
    return A_opt, S_opt


# In[114]:


O = np.ones((3, 1))


# In[ ]:





# # Saraga MyCode

# In[128]:


import os
import librosa as lb
import soundfile as sf
import museval
from matplotlib import pyplot as plt
from tqdm import tqdm
import librosa.display


# In[129]:


path = "/home/rajesh/Desktop/"
pathk = "/home/rajesh/Desktop/Saraga/KAMIR/"


# In[130]:


vocals, fs = lb.load(path+'vocals.wav')
mridangam, fs = lb.load(path+'mridangam.wav')
violin, fs = lb.load(path+'violin.wav')

mixture, fs = lb.load(pathk+'arti_mixture.wav')

X = np.array([vocals, mridangam, violin])
m = np.array([mixture])


# In[131]:


bleed_mixture = np.array([vocals + violin + mridangam])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 5 April Test

# In[ ]:


tvocals, fs = lb.load(path+'vocals.wav')
tmridangam, fs = lb.load(path+'mridangam.wav')
tviolin, fs = lb.load(path+'violin.wav')


S = np.array([vocals, mridangam, violin])


# In[132]:


True_Lambda = np.array([[0.719809, 0.195793, 0.0871341],
                        [0.057139, 0.889341, 0.078086],
                        [0.06324, 0.208965, 0.737475]])


# In[133]:


S_out_tl = np.linalg.inv(True_Lambda) @ X


# In[134]:


sf.write(path+'Avocals_pred.wav', S_out_tl[0], fs)
sf.write(path+'Amridangam_pred.wav', S_out_tl[1], fs)
sf.write(path+'Aviolin_pred.wav', S_out_tl[2], fs)


# In[137]:


vstft = lb.stft(vocals)
mstft = lb.stft(mridangam)
vistft = lb.stft(violin)


# In[142]:


m, n = vstft.shape


# In[138]:


Xf = np.vstack([np.abs(vstft), np.abs(mstft), np.abs(vistft)])


# In[186]:


a11 = np.full((m, m), 0.719809)
a12 = np.full((m, m), 0.195793)
a13 = np.full((m, m), 0.0871341)

a21 = np.full((m, m), 0.057139)
a22 = np.full((m, m), 0.889341)
a23 = np.full((m, m), 0.078086)

a31 = np.full((m, m), 0.06324)
a32 = np.full((m, m), 0.208965)
a33 = np.full((m, m), 0.737475)


# In[187]:


r1 = np.hstack([a11, a12, a13])
r2 = np.hstack([a21, a22, a23])
r3 = np.hstack([a31, a32, a33])


# In[188]:


Lambda = np.vstack([r1, r2, r3])


# In[189]:


Xf.shape, Lambda.shape


# In[193]:


Lambda_inv = np.linalg.inv(Lambda+0.00001*np.eye(3*m, 3*m))


# In[194]:


Sf = Lambda_inv @ Xf


# In[195]:


Sf.shape


# In[196]:


S1 = Sf[0:m, :]
S2 = Sf[m:m+m, :]
S3 = Sf[m+m:m+m+m, :]


# In[197]:


vphase = np.angle(vstft)
mphase = np.angle(mstft)
viphase = np.angle(vistft)


# In[198]:


S1.shape, S2.shape, S3.shape


# In[199]:


import cmath
def convert_complex(mag, phase):
    m, n = mag.shape
    complexx = []
    for i in range(m):
        for j in range(n):
            complexx.append(cmath.rect(mag[i][j], phase[i][j]))
    return lb.istft(np.array(complexx).reshape(m, n))


# In[200]:


vocal_pred = convert_complex(S1, vphase)
mridangam_pred = convert_complex(S2, mphase)
violin_pred = convert_complex(S3, viphase)


# In[201]:


sf.write(path+'lvocals_pred.wav', vocal_pred, fs)
sf.write(path+'lmridangam_pred.wav', mridangam_pred, fs)
sf.write(path+'lviolin_pred.wav', violin_pred, fs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[83]:


#O = np.array([[0.83382577, 1.5030332 , 0.05593537]]).T
#O.shape


# In[124]:


block = 25
A_pred, S_pred = [], []

for i in tqdm(range(0, X.shape[1], block)):
    x_ = X[:, i:i+block]
    m_ = bleed_mixture[:, i:i+block]
    A_opt, S_opt = Minimize(x_, m_, O, 0, 1, tol=1e-6, max_iter=1000)
    S_pred.append(S_opt)
    A_pred.append(A_opt)
    
S_pred = np.hstack(S_pred)


# In[125]:


A_avg = np.mean(A_pred, axis=0)
A_avg


# In[126]:


S_pred.shape


# In[87]:


SP = np.linalg.inv(A_avg) @ X


# In[127]:


sf.write(path+'Avocals_pred.wav', S_pred[0], fs)
sf.write(path+'Amridangam_pred.wav', S_pred[1], fs)
sf.write(path+'Aviolin_pred.wav', S_pred[2], fs)


# In[89]:


sf.write(path+'vocals_pred.wav', SP[0], fs)
sf.write(path+'mridangam_pred.wav', SP[1], fs)
sf.write(path+'violin_pred.wav', SP[2], fs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[90]:


stft_v = lb.stft(vocals)
stft_mr = lb.stft(mridangam)
stft_vi = lb.stft(violin)


stft_vp = lb.stft(S_pred[0])
stft_mrp = lb.stft(S_pred[1])
stft_vip = lb.stft(S_pred[2])


# In[91]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_v), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Bleed')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[92]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_vp), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Predicted')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[93]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_mr), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Mridgangam Bleed')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[94]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_mrp), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Mridangam Predicted')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[95]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_vi), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Violin Bleed')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[96]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(stft_vip), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Violin Predicted')
fig.colorbar(img, ax=ax, format="%+2.f dB")


# In[97]:


np.mean(A_pred, axis=0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SDR, SPECT< LAMBDA CALCULATION OF KAMIR

# In[276]:


name = 'Leaf - Wicked/'
path = ''+name
p = "/home/rajesh/Desktop/EXPERIMENT RESULTS/KAMIR/"
clean_path = "/home/rajesh/Desktop/Datasets/musdb18hq/train/"+name
i = name


# In[277]:


b1 = fs*60
b2 = int(fs*10+b1)


# In[278]:


bv, fs = lb.load(p+i+'/vocal.wav')
bb, fs = lb.load(p+i+'/bass.wav')
bd, fs = lb.load(p+i+'/drums.wav')
bo, fs = lb.load(p+i+'/others.wav')

tv, fs = lb.load(clean_path+'/vocals.wav')
tb, fs = lb.load(clean_path+'/bass.wav')
td, fs = lb.load(clean_path+'/drums.wav')
to, fs = lb.load(clean_path+'/other.wav')

tv = tv[b1:b2]
tb = tb[b1:b2]
td = td[b1:b2]
to = to[b1:b2]

pv, fs = lb.load(p+i+'/vocal_pred.wav')
pb, fs = lb.load(p+i+'/bass_pred.wav')
pd, fs = lb.load(p+i+'/drums_pred.wav')
po, fs = lb.load(p+i+'/others_pred.wav')


# In[ ]:





# In[ ]:





# In[279]:


v_sdr = compute_sdr(tv, pv, fs)
b_sdr = compute_sdr(tb, pb, fs)
d_sdr = compute_sdr(td, pd, fs)
o_sdr = compute_sdr(to, po, fs)

sdr = (v_sdr + b_sdr + d_sdr + o_sdr)/4
sdr, v_sdr, b_sdr, d_sdr, o_sdr


# In[280]:


X = np.array([bv, bb, bd, bo])
S = np.array([pv, pb, pd, po])


# In[281]:


A_pred = X @ np.linalg.pinv(S)
A_pred


# In[282]:


X_true = np.array([tv, tb, td, to])
A_true = X_true @ np.linalg.pinv(S)
A_true


# In[283]:


np.linalg.norm(A_true-A_pred)**2


# In[284]:


import numpy as np
import matplotlib.pyplot as plt

plt.matshow(A_pred)
plt.colorbar()
plt.title('Average Bleed Matrix')
plt.show()


# In[285]:


bvstft = lb.stft(bv)

pvstft = lb.stft(pv)

vstft = lb.stft(tv)


# In[286]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(bvstft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Vocal Bleed')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(p+i+'VocalBleedSpect.png')


# In[287]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(pvstft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Vocal Predicted')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(p+i+'VocalPredSpect.png')


# In[288]:


fig, ax = plt.subplots()
img = librosa.display.specshow(np.abs(vstft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Vocal True')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(p+i+'VocalTrueSpect.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # NN With Spect Input

# In[206]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, MaxPooling3D, UpSampling3D, Dense, Conv3D, Conv3DTranspose, Flatten, Reshape


# In[217]:


sources = 3
m = 1025
n = 431

inp = Input(shape=(sources, m, n, 1))
batch_size = 16
epochs = 100


# In[ ]:


l1 = Conv3D(128, (1, 3, 3), activation='relu')(inp)
l1 =  MaxPooling3D((1, 2, 2))(l1)
print(l1.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




