import numpy as np
import librosa as lb
import soundfile as sf
from matplotlib import pyplot as plt
from tqdm import tqdm
import museval
import os
import pandas as pd


def get_metrics(y):
    avg_y = []
    for i in range(len(y)):
        x = y[~np.isnan(y)]
        avg = sum(x)/len(x)
        avg_y.append(avg)
    return avg_y

def compute_sdr(true, reconstructed, fs):
    t = np.array([true])
    r = np.array([reconstructed])

    sdr, isr, sir, sar = museval.evaluate(t, r, win=fs, hop=fs)
        
    avg_sdr = get_metrics(sdr)
    avg_isr = get_metrics(isr) #Source to Spatial Distortion Image
    avg_sir = get_metrics(sir)
    avg_sar = get_metrics(sar)

    return sum(avg_sdr)/len(avg_sdr)


def blockleastnormsoln(r, s):
    r = r.flatten()
    s = s.flatten()
    b = r-s
    
    val = int(r.shape[0]/4)
    s0 = s[0:val]
    s1 = s[val:2*val]
    s2 = s[2*val:3*val]
    s3 = s[3*val:4*val]
    
    z = np.zeros(val)

    c1 = np.hstack((s1, z, z, z))
    c2 = np.hstack((s2, z, z, z))
    c3 = np.hstack((s3, z, z, z))

    c4 = np.hstack((z, s0, z, z))
    c5 = np.hstack((z, s2, z, z))
    c6 = np.hstack((z, s3, z, z))

    c7 = np.hstack((z, z, s0, z))
    c8 = np.hstack((z, z, s1, z))
    c9 = np.hstack((z, z, s3, z))

    c10 = np.hstack((z, z, z, s0))
    c11 = np.hstack((z, z, z, s1))
    c12 = np.hstack((z, z, z, s2))
    
    A = np.stack((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12), axis=1)
    
    soln, residual, rank, singular = np.linalg.lstsq(A, b, rcond=-1)
    
    return np.array([[1, soln[0], soln[1], soln[2]],
                     [soln[3], 1, soln[4], soln[5]],
                     [soln[6], soln[7], 1, soln[8]],
                     [soln[9], soln[10], soln[11], 1]])




bleed_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/train/'
clean_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq/train/'
out_path = '/home/anchal/Desktop/rajesh/factorisation/'


files = sorted(os.listdir(clean_path))

lmbda = []
print('Started')
k = 1
for i in files:

    print('Processing: ',i)

    b_path = bleed_path+i
    c_path = clean_path+i

    bvocals, fs = lb.load(b_path+'/vocals.wav')
    bbass, fs = lb.load(b_path+'/bass.wav')
    bdrums, fs = lb.load(b_path+'/drums.wav')
    bother, fs = lb.load(b_path+'/other.wav')

    vocals, fs = lb.load(c_path+'/vocals.wav')
    bass, fs = lb.load(c_path+'/bass.wav')
    drums, fs = lb.load(c_path+'/drums.wav')
    other, fs = lb.load(c_path+'/other.wav')

    if len(bbass) > len(bass):
        n = len(bass)
    else:
        n = len(bbass)

    R = np.array([bvocals[:n], bbass[:n], bdrums[:n], bother[:n]])
    S = np.array([vocals[:n], bass[:n], drums[:n], other[:n]])

    print('Inverting....')
    
    samples = R.shape[1] #use seconds or samples
    block = samples

    block_lamda = []
    for i in range(0, R.shape[1], block):
        r = R[: ,i:i+block]
        s = S[: ,i:i+block]
        block_lamda.append(blockleastnormsoln(r, s))


    lmbda.append(block_lamda[0])
   
    print(k, '/100 files processed..')
    print('-----')
    k+=1

np.save(out_path+'org_lambda.npy', lmbda)