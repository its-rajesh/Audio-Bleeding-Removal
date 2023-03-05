import numpy as np
import librosa as lb
import soundfile as sf
from rpca import RPCA
import cmath
import museval


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




bleed_path = '/home/rajesh/Desktop/Datasets/musdb18hq_bleeded/train/Music Delta - Hendrix/'
bvocals, fs = lb.load(bleed_path+'vocals.wav')
bbass, fs = lb.load(bleed_path+'bass.wav')
bdrums, fs = lb.load(bleed_path+'drums.wav')
bother, fs = lb.load(bleed_path+'other.wav')


clean_path = '/home/rajesh/Desktop/Datasets/musdb18hq/train/Music Delta - Hendrix/'
vocals, fs = lb.load(clean_path+'vocals.wav')
bass, fs = lb.load(clean_path+'bass.wav')
drums, fs = lb.load(clean_path+'drums.wav')
other, fs = lb.load(clean_path+'other.wav')


stft_bvocals = lb.stft(bvocals)

L, S = RPCA.robust_pca(RPCA, np.abs(stft_bvocals))

print(np.abs(stft_bvocals), L, S)
print(np.abs(stft_bvocals).shape, L.shape, S.shape)

phase = np.angle(stft_bvocals)
bg_estimate = calc_istft(L, phase)
fg_estimate = calc_istft(S, phase)

f_sdr = compute_sdr(vocals, fg_estimate, fs)
b_sdr = compute_sdr(vocals, bg_estimate, fs)
print(f_sdr, b_sdr)


sf.write('/home/rajesh/Desktop/fg_estimate.wav', fg_estimate, fs)
sf.write('/home/rajesh/Desktop/bg_estimate.wav', bg_estimate, fs)


