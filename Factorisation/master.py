import numpy as np
import librosa as lb
import soundfile as sf
from matplotlib import pyplot as plt
from tqdm import tqdm
import museval
import os
import pandas as pd


bleed_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/test/'
clean_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/'
out_path = '/home/anchal/Desktop/rajesh/factorisation/'

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

def train(files, A):

    sdr = []
    v_sdr, b_sdr, d_sdr, o_sdr = [], [], [], []
    print('Started')

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

        print('Predicting....')
        A_inv = np.linalg.inv(A)
        s = np.dot(A_inv, R)
        print('Done..!')

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

        print('-----')


    sdr_file = pd.DataFrame(
        {'vocal': v_sdr,
        'bass': b_sdr,
        'drums': d_sdr,
        'others': o_sdr,
        'overall': sdr
        })
    
    return sdr_file

   
Amin = np.array([[1.        , 0.10000096, 0.10000116, 0.10000022],
        [0.10000026, 1.        , 0.10000087, 0.10000014],
        [0.10000026, 0.10000102, 1.        , 0.10000016],
        [0.10000038, 0.10000105, 0.10000071, 1.        ]])

Amax = np.array([[1.        , 0.10000096, 0.10000116, 0.10000022],
        [0.10000026, 1.        , 0.10000087, 0.10000014],
        [0.10000026, 0.10000102, 1.        , 0.10000016],
        [0.10000038, 0.10000105, 0.10000071, 1.        ]])

Aavg = np.array([[1.        , 0.10000096, 0.10000116, 0.10000022],
        [0.10000026, 1.        , 0.10000087, 0.10000014],
        [0.10000026, 0.10000102, 1.        , 0.10000016],
        [0.10000038, 0.10000105, 0.10000071, 1.        ]])

Amedian = np.array([[1.        , 0.10000096, 0.10000116, 0.10000022],
        [0.10000026, 1.        , 0.10000087, 0.10000014],
        [0.10000026, 0.10000102, 1.        , 0.10000016],
        [0.10000038, 0.10000105, 0.10000071, 1.        ]])


files = sorted(os.listdir(clean_path))
print('-----------------------------------------------')
print('Minimum Lambda..')
print('-----------------------------------------------')
min_sdr = train(files, Amin)
print('-----------------------------------------------')
print('Maximum Lambda..')
print('-----------------------------------------------')
max_sdr = train(files, Amax)
print('-----------------------------------------------')
print('Average Lambda..')
print('-----------------------------------------------')
avg_sdr = train(files, Aavg)
print('-----------------------------------------------')
print('Median Lambda..')
print('-----------------------------------------------')
median_sdr = train(files, Amedian)


min_sdr.to_csv(out_path+'min_sdr.csv')
max_sdr.to_csv(out_path+'max_sdr.csv')
avg_sdr.to_csv(out_path+'avg_sdr.csv')
median_sdr.to_csv(out_path+'median_sdr.csv')