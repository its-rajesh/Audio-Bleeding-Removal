import os
import numpy as np
import librosa as lb
import soundfile as sf
from tqdm import tqdm


def gen_hardmask(bleed, pred, tr=0.002): #0.9

    vdiff = np.abs(bleed) - np.abs(pred)
    print(np.mean(vdiff), np.median(vdiff), np.min(vdiff), np.max(vdiff))
    vmask = vdiff > tr
    vft = np.abs(bleed) * vmask

    return vft

def replaceValues(source_matrix, replacement_matrix):
    # Find indices of zeros in the source_matrix
    zero_indices = np.argwhere(source_matrix == 0)

    # Replace zeros with corresponding values from replacement_matrix
    for row, col in zero_indices:
        source_matrix[row, col] = replacement_matrix[row, col]

    return source_matrix

path = "/home/rajesh/Desktop/Clean/Audio Results/"

bleed = 'Saraga/'

for model in ['dfUNet/', 'dfUNet-GAT/']:
    dpath = path+bleed+model
    folders = sorted(os.listdir(dpath))

    for fold in tqdm(folders):
        subpath = dpath+fold+'/'

        dbvocal, fs = lb.load(subpath+'bvocal.wav')
        dbmridangam, fs = lb.load(subpath+'bmridangam.wav')
        dbviolin, fs = lb.load(subpath+'bviolin.wav')
        dbmix, fs = lb.load(subpath+'bother.wav')

        dpvocal, fs = lb.load(subpath+'pvocal.wav')
        dpmridangam, fs = lb.load(subpath+'pmridangam.wav')
        dpviolin, fs = lb.load(subpath+'pviolin.wav')
        dpmix, fs = lb.load(subpath+'pother.wav')

        dbv = lb.stft(dbvocal)
        dbm = lb.stft(dbmridangam)
        dbvi = lb.stft(dbviolin)
        dbmx = lb.stft(dbmix)

        dpv = lb.stft(dpvocal)
        dpm = lb.stft(dpmridangam)
        dpvi = lb.stft(dpviolin)
        dpmx = lb.stft(dpmix)

        # Hard Mask
        vft = gen_hardmask(dbv, dpv)
        bft = gen_hardmask(dbm, dpm)
        dft = gen_hardmask(dbvi, dpvi)
        oft = gen_hardmask(dbmx, dpmx)

        # Spectral subtraction & writting
        ppv = (0.5 * np.abs(dpv) - dft - bft - oft) + vft
        ppv[ppv < 0] = 0
        ppv = replaceValues(ppv, 0.1 * np.abs(dpv))
        reconst = ppv * np.angle(dpv) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/mvocal.wav', reconst, fs)

        ppm = (0.5 * np.abs(dpm) - vft - bft - oft) + dft
        ppm[ppm < 0] = 0
        ppm = replaceValues(ppm, 0.1 * np.abs(dpm))
        reconst = ppm * np.angle(dpm) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/mmridangam.wav', reconst, fs)

        ppvi = (0.5 * np.abs(dpvi) - dft - vft - oft) + bft
        ppvi[ppvi < 0] = 0
        ppvi = replaceValues(ppvi, 0.1 * np.abs(dpvi))
        reconst = ppvi * np.angle(dpvi) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/mviolin.wav', reconst, fs)

        ppo = (0.5 * np.abs(dpmx) - dft - vft - bft) + oft
        ppo[ppo < 0] = 0
        ppo = replaceValues(ppo, 0.1 * np.abs(dpmx))
        reconst = ppo * np.angle(dpmx) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/mother.wav', reconst, fs)

       




