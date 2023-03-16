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



bleed_path = '/home/anchal/Desktop/rajesh/Datasets/sample_saraga/'
out_path = '/home/anchal/Desktop/rajesh/factorisation/SaragaOutputs/'


files = sorted(os.listdir(bleed_path))

sdr, bsdr = [], []
v_sdr, b_sdr, d_sdr, o_sdr = [], [], [], []
bv_sdr, bb_sdr, bd_sdr, bo_sdr = [], [], [], []
lmda = []
print('Started')

for i in tqdm(files):

    print('Processing: ',i)

    b_path = bleed_path+i

    bvocals, fs = lb.load(b_path+'/vocals.wav')
    bmridangam, fs = lb.load(b_path+'/mridangam.wav')
    bviolin, fs = lb.load(b_path+'/violin.wav')
    

    R = np.array([bvocals, bmridangam, bviolin])

    print('Factorising....')
    gamma1 = 0.0005
    gamma2 = 0.2
    A, s = vector_factorisation(R, gamma1, gamma2)
    print('Factorized!')
    lmda.append(A)

    os.makedirs(out_path+i, mode = 0o777, exist_ok = False)

    print('Writting Files...')
    sf.write(out_path+i+'/pred_vocals.wav', s[0], fs)
    sf.write(out_path+i+'/pred_mridangam.wav', s[0], fs)
    sf.write(out_path+i+'/pred_violin.wav', s[0], fs)
    print('-----------')

    

np.save('/home/anchal/Desktop/rajesh/factorisation/saraga_lambda.npy')

