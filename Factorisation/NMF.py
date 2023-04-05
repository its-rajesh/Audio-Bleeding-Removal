import numpy as np


def objective(X, A, S):
    return np.linalg.norm(X-(A@S))**2


def updateA(X, A, S):
    regulariser = 0.00001
    XST = X @ S.T 
    SST = S @ S.T
    ASST = A @ SST
    m, n = ASST.shape
    ASST = ASST + regulariser * np.ones((m, n)) # To prevent divide by zero
    return np.multiply(A, np.divide(XST, ASST))


def updateS(X, A, S):
    regulariser = 0.00001
    ATX = A.T @ X 
    ATA = A.T @ A 
    ATAS = ATA @ S
    m, n = ATAS.shape
    ATAS = ATAS  + regulariser * np.ones((m, n)) # To prevent divide by zero
    return np.multiply(S, np.divide(ATX, ATAS))


def Minimize(X, tol=1e-3, max_iter=1000):
    n, l = X.shape
    A_opt = np.eye(n)
    S_opt = X.copy()
    #A = np.random.random((n, n))
    #S_opt = np.random.random((n, l))

    iters = 0
    while True:
        
        A_opt = updateA(X, A_opt, S_opt)
        S_opt = updateS(X, A_opt, S_opt)
        loss = objective(X, A_opt, S_opt)
        #print('ITERATION:', iters+1, 'LOSS:', loss, sep=' ')
        if iters>0 and (prev_loss - loss) <= tol:
            break
        prev_loss = loss
        iters +=1
        if iters >= max_iter:
            print('MAXIMUM ITERATION REACHED!!!')
            break
        
    return A_opt, S_opt

##################################################################################
#                               TOY EXAMPLE                                      #
##################################################################################

S = np.random.random((3, 10))
A = np.array([[1, 0.1, 0.2],
              [0.12, 1, 0.12],
              [0.21, 0.15, 1]])
X = A @ S

A_toy, S_toy = Minimize(X, tol=1e-3, max_iter=1000)




##################################################################################
#                               MUSDB18HQ TEST                                   #
##################################################################################


import numpy as np
import os
import librosa as lb
import soundfile as sf
import museval
from matplotlib import pyplot as plt
from tqdm import tqdm
import librosa.display


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


def spectra(X, S, outpath):
    
    stft_bv = lb.stft(X[0])
    stft_bb = lb.stft(X[1])
    stft_bd = lb.stft(X[2])
    stft_bo = lb.stft(X[3])
    
    stft_vp = lb.stft(S[0])
    stft_bp = lb.stft(S[1])
    stft_dp = lb.stft(S[2])
    stft_op = lb.stft(S[3])
    
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(np.abs(stft_vp), x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Vocal Predicted')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    #plt.savefig(outpath+'VocalpredictedSpect.png')
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(np.abs(stft_bv), x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Vocal Bleed')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    #plt.savefig(outpath+'VocalBleedSpect.png')


def bleed_remove(clean_path, outpath):
    vocals, fs = lb.load(clean_path+'tvocals.wav')
    bass, fs = lb.load(clean_path+'tbass.wav')
    drums, fs = lb.load(clean_path+'tdrums.wav')
    other, fs = lb.load(clean_path+'tother.wav')
    
    bvocals, fs = lb.load(clean_path+'bvocals.wav')
    bbass, fs = lb.load(clean_path+'bbass.wav')
    bdrums, fs = lb.load(clean_path+'bdrums.wav')
    bother, fs = lb.load(clean_path+'bother.wav')

    X = np.array([bvocals, bbass, bdrums, bother])
    S = np.array([vocals, bass, drums, other])
    
    
    block = 25
    A_pred, S_pred = [], []

    for i in tqdm(range(0, X.shape[1], block)):
        x_ = X[:, i:i+block]
        A_opt, S_opt = Minimize(x_, tol=1e-6)
        S_pred.append(S_opt)
        A_pred.append(A_opt)

    S_pred = np.hstack(S_pred)
    
    sf.write(outpath+'pred_vocal.wav', S_pred[0], fs)
    sf.write(outpath+'pred_bass.wav', S_pred[1], fs)
    sf.write(outpath+'pred_drums.wav', S_pred[2], fs)
    sf.write(outpath+'pred_others.wav', S_pred[3], fs)
    
    tv_sdr = compute_sdr(vocals, bvocals, fs)
    tb_sdr = compute_sdr(bass, bbass, fs)
    td_sdr = compute_sdr(drums, bdrums, fs)
    to_sdr = compute_sdr(other, bother, fs)

    true_sdr = (tv_sdr + tb_sdr + td_sdr + to_sdr)/4
    print(true_sdr, tv_sdr, tb_sdr, td_sdr, to_sdr)
    
    v_sdr = compute_sdr(vocals, S_pred[0], fs)
    b_sdr = compute_sdr(bass, S_pred[1], fs)
    d_sdr = compute_sdr(drums, S_pred[2], fs)
    o_sdr = compute_sdr(other, S_pred[3], fs)

    obtained_sdr = (v_sdr + b_sdr + d_sdr + o_sdr)/4
    print(obtained_sdr, v_sdr, b_sdr, d_sdr, o_sdr)
    
    A_avg = np.average(A_pred, axis=0)
    print(np.round(A_avg, 3))
    
    A_act = X @ np.linalg.pinv(S)
    print(np.round(A_act, 3))
    print(np.linalg.norm(A_act - A_avg)**2)
    
    spectra(X, S, outpath)
    
    return np.array(A_pred), S_pred


##################################################################################
#                            MAIN DRIVER CODE                                    #
##################################################################################

clean_path = "/home/rajesh/Desktop/"
outpath = "/home/rajesh/Desktop/"


A_pred, S_pred = bleed_remove(clean_path, outpath)

print(A_pred)
print(S_pred)

plt.plot(S_pred[0])
plt.show()



