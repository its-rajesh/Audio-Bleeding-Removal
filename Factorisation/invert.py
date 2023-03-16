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



bleed_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/test/'
clean_path = '/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/'
out_path = '/home/anchal/Desktop/rajesh/factorisation/Outputs/'


files = sorted(os.listdir(clean_path))

sdr, bsdr = [], []
v_sdr, b_sdr, d_sdr, o_sdr = [], [], [], []
bv_sdr, bb_sdr, bd_sdr, bo_sdr = [], [], [], []
lmda = []
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
    S = np.array([vocals[:n], bass[:n], drums[:n], other[:n]])

    print('Calculating Inverse....')
    A = np.array([[1.        , 0.10000096, 0.10000116, 0.10000022],
        [0.10000026, 1.        , 0.10000087, 0.10000014],
        [0.10000026, 0.10000102, 1.        , 0.10000016],
        [0.10000038, 0.10000105, 0.10000071, 1.        ]])
    

    s = np.linalg.lstsq(A, R, rcond=None)[0]
    print('Inverse Done!')

    os.makedirs(out_path+i, mode = 0o777, exist_ok = False)

    print('Writting Files...')
    sf.write(out_path+i+'/pred_vocals.wav', s[0], fs)
    sf.write(out_path+i+'/pred_bass.wav', s[0], fs)
    sf.write(out_path+i+'/pred_drums.wav', s[0], fs)
    sf.write(out_path+i+'/pred_other.wav', s[0], fs)

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


sdr_file.to_csv('/home/anchal/Desktop/rajesh/factorisation/sdr.csv')
