import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, MaxPooling2D
from matplotlib import pyplot as plt
from tqdm import tqdm
import soundfile as sf
import museval
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

    return sum(avg_sdr)/len(avg_sdr), sum(avg_isr)/len(avg_isr), sum(avg_sir)/len(avg_sir), sum(avg_sar)/len(avg_sar)


print('Loading Files...')
path = "/home/anchal/Desktop/rajesh/Datasets/extensive_bleed/"

outpath = "/home/anchal/Desktop/rajesh/NN_CBR/OVERALL/"

X_test = np.load(path+'Xtest.npy')
Y_test = np.load(path+'Ytest.npy')


print(X_test.shape)

'''
sf.write(path+'/Outputs/bvocal.wav', X_test[0][0], 22050)
sf.write(path+'/Outputs/bbass.wav', X_test[0][1], 22050)
sf.write(path+'/Outputs/bdrums.wav', X_test[0][2], 22050)
sf.write(path+'/Outputs/bother.wav', X_test[0][3], 22050)

sf.write(path+'/Outputs/tvocal.wav', Y_test[0][0], 22050)
sf.write(path+'/Outputs/tbass.wav', Y_test[0][1], 22050)
sf.write(path+'/Outputs/tdrums.wav', Y_test[0][2], 22050)
sf.write(path+'/Outputs/tother.wav', Y_test[0][3], 22050)
'''

print("Evaluating..")

model = tf.keras.models.load_model('NNCr.h5')

predictions = model.predict(X_test)
#print(predictions)

fs = 22050
Atv_sdr, Atv_isr, Atv_sir, Atv_sar = [], [], [], []
Atb_sdr, Atb_isr, Atb_sir, Atb_sar = [], [], [], []
Atd_sdr, Atd_isr, Atd_sir, Atd_sar = [], [], [], []
Ato_sdr, Ato_isr, Ato_sir, Ato_sar = [], [], [], []

Av_sdr, Av_isr, Av_sir, Av_sar = [], [], [], []
Ab_sdr, Ab_isr, Ab_sir, Ab_sar = [], [], [], []
Ad_sdr, Ad_isr, Ad_sir, Ad_sar = [], [], [], []
Ao_sdr, Ao_isr, Ao_sir, Ao_sar = [], [], [], []

Atrue_sdr, Atrue_isr, Atrue_sir, Atrue_sar = [], [], [], []
Aobtained_sdr, Aobtained_isr, Aobtained_sir, Aobtained_sar = [], [], [], []
L2norm = []

for i in tqdm(range(X_test.shape[0])):

    A = predictions[i]
    A_act = X_test[i] @ np.linalg.pinv(Y_test[i])
    closeness = np.linalg.norm(A-A_act)**2
    L2norm.append(closeness)

    S_pred = np.linalg.inv(A) @ X_test[i]

    tv_sdr, tv_isr, tv_sir, tv_sar = compute_sdr(Y_test[i][0], X_test[i][0], fs)
    tb_sdr, tb_isr, tb_sir, tb_sar = compute_sdr(Y_test[i][1], X_test[i][1], fs)
    td_sdr, td_isr, td_sir, td_sar = compute_sdr(Y_test[i][2], X_test[i][2], fs)
    to_sdr, to_isr, to_sir, to_sar = compute_sdr(Y_test[i][3], X_test[i][3], fs)

    true_sdr = (tv_sdr + tb_sdr + td_sdr + to_sdr)/4
    true_isr = (tv_isr + tb_isr + td_isr + to_isr)/4
    true_sir = (tv_sir + tb_sir + td_sir + to_sir)/4
    true_sar = (tv_sar + tb_sar + td_sar + to_sar)/4
    #print(true_sdr, tv_sdr, tb_sdr, td_sdr, to_sdr)

    v_sdr, v_isr, v_sir, v_sar = compute_sdr(Y_test[i][0], S_pred[0], fs)
    b_sdr, b_isr, b_sir, b_sar = compute_sdr(Y_test[i][1], S_pred[1], fs)
    d_sdr, d_isr, d_sir, d_sar = compute_sdr(Y_test[i][2], S_pred[2], fs)
    o_sdr, o_isr, o_sir, o_sar = compute_sdr(Y_test[i][3], S_pred[3], fs)

    obtained_sdr = (v_sdr + b_sdr + d_sdr + o_sdr)/4
    obtained_isr = (v_isr + b_isr + d_isr + o_isr)/4
    obtained_sir = (v_sir + b_sir + d_sir + o_sir)/4
    obtained_sar = (v_sar + b_sar + d_sar + o_sar)/4
    #print(obtained_sdr, v_sdr, b_sdr, d_sdr, o_sdr)

    Atv_sdr.append(tv_sdr)
    Atv_isr.append(tv_isr)
    Atv_sir.append(tv_sir)
    Atv_sar.append(tv_sar)

    Atb_sdr.append(tb_sdr)
    Atb_isr.append(tb_isr)
    Atb_sir.append(tb_sir)
    Atb_sar.append(tb_sar)

    Atd_sdr.append(td_sdr)
    Atd_isr.append(td_isr)
    Atd_sir.append(td_sir)
    Atd_sar.append(td_sar)

    Ato_sdr.append(to_sdr)
    Ato_isr.append(to_isr)
    Ato_sir.append(to_sir)
    Ato_sar.append(to_sar)

    Av_sdr.append(v_sdr)
    Av_isr.append(v_isr)
    Av_sir.append(v_sir)
    Av_sar.append(v_sar)

    Ab_sdr.append(b_sdr)
    Ab_isr.append(b_isr)
    Ab_sir.append(b_sir)
    Ab_sar.append(b_sar)

    Ad_sdr.append(d_sdr)
    Ad_isr.append(d_isr)
    Ad_sir.append(d_sir)
    Ad_sar.append(d_sar)

    Ao_sdr.append(o_sdr)
    Ao_isr.append(o_isr)
    Ao_sir.append(o_sir)
    Ao_sar.append(o_sar)

    Atrue_sdr.append(true_sdr)
    Atrue_isr.append(true_isr)
    Atrue_sir.append(true_sir)
    Atrue_sar.append(true_sar)

    Aobtained_sdr.append(obtained_sdr)
    Aobtained_isr.append(obtained_isr)
    Aobtained_sir.append(obtained_sir)
    Aobtained_sar.append(obtained_sar)


    

sdr = pd.DataFrame(
    {'vocal sdr': Atv_sdr,
     'bass sdr': Atb_sdr,
     'drums sdr': Atd_sdr,
     'other sdr': Ato_sdr,
     'Overall sdr': Atrue_sdr,
     'vocal isr': Atv_isr,
     'bass isr': Atb_isr,
     'drums isr': Atd_isr,
     'other isr': Ato_isr,
     'Overall isr': Atrue_isr,
     'vocal sir': Atv_sir,
     'bass sir': Atb_sir,
     'drums sir': Atd_sir,
     'other sir': Ato_sir,
     'Overall sir': Atrue_sir,
     'vocal sar': Atv_sar,
     'bass sar': Atb_sar,
     'drums sar': Atd_sar,
     'other sar': Ato_sar,
     'Overall sar': Atrue_sar,
     'Pred vocal sdr': Av_sdr,
     'Pred bass sdr': Ab_sdr,
     'Pred drums sdr': Ad_sdr,
     'Pred other sdr': Ao_sdr,
     'Pred Overall sdr': Aobtained_sdr,
     'Pred vocal isr': Av_isr,
     'Pred bass isr': Ab_isr,
     'Pred drums isr': Ad_isr,
     'Pred other isr': Ao_isr,
     'Pred Overall isr': Aobtained_isr,
     'Pred vocal sir': Av_sir,
     'Pred bass sir': Ab_sir,
     'Pred drums sir': Ad_sir,
     'Pred other sir': Ao_sir,
     'Pred Overall sir': Aobtained_sir,
     'Pred vocal sar': Av_sar,
     'Pred bass sar': Ab_sar,
     'Pred drums sar': Ad_sar,
     'Pred other sar': Ao_sar,
     'Pred Overall sar': Aobtained_sar,
     'Closeness': L2norm})

sdr.to_csv('sdr.csv')

'''
sf.write(path+'/Outputs/pvocal.wav', S_pred[0], 22050)
sf.write(path+'/Outputs/pbass.wav', S_pred[1], 22050)
sf.write(path+'/Outputs/pdrums.wav', S_pred[2], 22050)
sf.write(path+'/Outputs/pother.wav', S_pred[3], 22050)
'''


