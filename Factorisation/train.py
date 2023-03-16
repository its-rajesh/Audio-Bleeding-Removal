import numpy as np
import librosa as lb
import soundfile as sf
from matplotlib import pyplot as plt
from tqdm import tqdm
import museval
import os
import pandas as pd


def vector_factorisation(x, gamma1, gamma2, max_iter=1000, tol=1e-50):
    n, l = x.shape
    s = np.copy(x)
    A = np.eye(n)
    epsilon = 1e-10

    for i in range(max_iter):
        # update A
        for j in range(n):
            A[j, j] = 1  # diagonal elements
            for k in range(n):
                if k != j:
                    num = x[j] @ s[k] - A[k] @ s + A[j, k] * s[j] * s[k]
                    A[j, k] = max(gamma1, min(gamma2, np.squeeze(num / (np.dot(s[j], s[j]) + epsilon))[0]))



        # update s
        s = np.linalg.lstsq(A, x, rcond=None)[0]

        # calculate error
        error = np.linalg.norm(x - A @ s) ** 2

        if i > 0 and abs(error - prev_error) < tol:
            break

        prev_error = error

    return A, s


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




bleed_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/train/'
clean_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq/train/'


files = sorted(os.listdir(clean_path))

sdr, bsdr = [], []
v_sdr, b_sdr, d_sdr, o_sdr = [], [], [], []
bv_sdr, bb_sdr, bd_sdr, bo_sdr = [], [], [], []
lmda = []
print('Started')

for i in tqdm(files):

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

    print('Factorising....')
    gamma1 = 0.0005
    gamma2 = 0.1
    A, s = vector_factorisation(R, gamma1, gamma2)
    print('Factorized!')
    lmda.append(A)

    print('Computing SDR...')
    q1 = compute_sdr(vocals[:n], s[0], fs)
    q2 = compute_sdr(bass[:n], s[1], fs)
    q3 = compute_sdr(drums[:n], s[2], fs)
    q4 = compute_sdr(other[:n], s[3], fs)

    q5 = (q1+q2+q3+q4)/4

    v_sdr.append(q1)
    b_sdr.append(q2)
    d_sdr.append(q3)
    o_sdr.append(q4)
    sdr.append(q5)

    print(sdr)
    

    z1 = compute_sdr(vocals[:n], bvocals[:n], fs)
    z2 = compute_sdr(bass[:n], bbass[:n], fs)
    z3 = compute_sdr(drums[:n], bdrums[:n], fs)
    z4 = compute_sdr(other[:n], bother[:n], fs)

    z5 = (z1 + z2 + z3 + z4)/4

    bv_sdr.append(z1)
    bb_sdr.append(z2)
    bd_sdr.append(z3)
    bo_sdr.append(z4)
    bsdr.append(z5)

    print(bsdr)
    print('-----')


sdr_file = pd.DataFrame(
    {'vocal': v_sdr,
     'bass': b_sdr,
     'drums': d_sdr,
     'others': o_sdr,
     'overall': sdr
    })

bsdr_file = pd.DataFrame(
    {'vocal': bv_sdr,
     'bass': bb_sdr,
     'drums': bd_sdr,
     'others': bo_sdr,
     'overall': bsdr
    })

np.save('/home/anchal/Desktop/rajesh/factorisation/lambda.npy')
sdr_file.to_csv('/home/anchal/Desktop/rajesh/factorisation/sdr.csv')
bsdr_file.to_csv('/home/anchal/Desktop/rajesh/factorisation/actual_sdr.csv')
