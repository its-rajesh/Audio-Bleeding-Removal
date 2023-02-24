####################################################################################################
#                                     OVERALL SDR COMPUTATION                                      #
####################################################################################################
# Author: Rajesh R (Git: its-rajesh)
# Date: 14/Januray/2023

import numpy as np
import os
import librosa as lb
from tensorflow import keras
import DynamicInput as dp
import cmath
from matplotlib import pyplot as plt
import museval
import soundfile as sf
from tqdm import tqdm
import pandas as pd



def calc_istft(mag, phase):
        m, n = mag.shape
        complexx = []
        for i in range(m):
            for j in range(n):
                complexx.append(cmath.rect(mag[i][j], phase[i][j]))

        r = np.array(complexx).reshape(m, n)
        return lb.istft(r)

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

    return sum(avg_sdr)/len(avg_sdr), sum(avg_isr)/len(avg_isr), sum(avg_sir)/len(avg_sir), sum(avg_sar)/len(avg_sar)

def iter1(di3_dca, bpath, tpath, out_path):

    xtest, fs = lb.load(bpath+'/other.wav')
    ytest, fs = lb.load(tpath+'/other.wav')

    stft = lb.stft(xtest)
    inp = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft), 3, 1)

    y_stft = lb.stft(ytest)
    ytest_inp = dp.DynamicInput.encode(dp.DynamicInput, np.abs(y_stft), 3, 1)
    loss = di3_dca.evaluate(inp,  ytest_inp, verbose=1)

    predictions = di3_dca.predict(inp, verbose=1)
    predictions = np.squeeze(predictions, axis=3)

    decoded = dp.DynamicInput.decode(dp.DynamicInput, predictions, 3, 1)
    decoded_y = dp.DynamicInput.decode(dp.DynamicInput, ytest_inp, 3, 1)

    reconstructed = calc_istft(decoded, np.angle(stft))
    true = calc_istft(decoded_y, np.angle(y_stft))

    os.makedirs(out_path, exist_ok=True)

    sf.write(out_path + '/reconstructed.wav', reconstructed, fs)
    sf.write(out_path + '/bleed.wav', xtest, fs)

    sdr_i, isr_i, sir_i, sar_i = compute_sdr(true, reconstructed, fs)

    return sdr_i, isr_i, sir_i, sar_i


##############################################################################################################
# DRIVER CODE
##############################################################################################################

bpath = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/test/"
tpath = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/"

print("Loading Model")
di3_dca = keras.models.load_model("/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Others/other_model.h5")


bfiles = sorted(os.listdir(bpath))
tfiles = sorted(os.listdir(tpath))

out_path = os.getcwd() + '/Results/'

print('CALCULATING SDR & WRITTING RESULTS...')

avg_sdr, avg_isr, avg_sir, avg_sar = [], [], [], []
n = len(os.listdir(bpath))
name = []
for i in tqdm(range(n)):
    try:
        sdr, isr, sir, sar = iter1(di3_dca, bpath+bfiles[i], tpath+tfiles[i], out_path+bfiles[i])
        avg_sdr.append(sdr)
        avg_isr.append(isr)
        avg_sir.append(sir)
        avg_sar.append(sar)
        name.append(bfiles[i])
    except:
        print("file Skipped")


print('----------------------------------------')
print('RESULT')
print('OVERALL SDR:', sum(avg_sdr)/len(avg_sdr))
print('OVERALL ISR:', sum(avg_isr)/len(avg_isr))
print('OVERALL SIR:', sum(avg_sir)/len(avg_sir))
print('OVERALL SAR:', sum(avg_sar)/len(avg_sar))
print('----------------------------------------')

print('WRITTING CSV FILE...')

res = pd.DataFrame({'files':name, 'sdr':avg_sdr, 'isr':avg_isr, 'sir':avg_sir, 'sar':avg_sar})
res.to_csv(out_path+'sdr.csv')

print('END')



