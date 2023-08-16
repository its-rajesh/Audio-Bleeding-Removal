import os
import numpy as np
import librosa as lb
import soundfile as sf
from tqdm import tqdm


def gen_hardmask(bleed, pred, tr=0.9):

    vdiff = np.abs(bleed) - np.abs(pred)
    vmask = vdiff > tr
    vft = np.abs(bleed) * vmask

    return vft


path = "/home/rajesh/Desktop/Masking/"

bleed = 'Saraga_Outputs/'

for model in ['dfUNet/', 'KAMIR/']:
    dpath = path+model+bleed
    folders = sorted(os.listdir(dpath))

    for fold in tqdm(folders):
        subpath = dpath+fold+'/'

        dbvocal, fs = lb.load(subpath+'bvocal.wav')
        dbmridangam, fs = lb.load(subpath+'bmridangam.wav')
        dbviolin, fs = lb.load(subpath+'bviolin.wav')
        #dbmix, fs = lb.load(subpath+'bother.wav')

        dpvocal, fs = lb.load(subpath+'pvocal.wav')
        dpmridangam, fs = lb.load(subpath+'pmridangam.wav')
        dpviolin, fs = lb.load(subpath+'pviolin.wav')
        #dpmix, fs = lb.load(subpath+'pother.wav')

        dbv = lb.stft(dbvocal)
        dbm = lb.stft(dbmridangam)
        dbvi = lb.stft(dbviolin)
        #dbmx = lb.stft(dbmix)

        dpv = lb.stft(dpvocal)
        dpm = lb.stft(dpmridangam)
        dpvi = lb.stft(dpviolin)
        #dpmx = lb.stft(dpmix)

        # Hard Mask
        vft = gen_hardmask(dbv, dpv)
        mft = gen_hardmask(dbm, dpm)
        vift = gen_hardmask(dbvi, dpvi)

        # Spectral subtraction & writting
        ppv = (np.abs(dpv) - mft - vift) + vft
        ppv[ppv < 0] = 0
        reconst = ppv * np.angle(dpv) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/svocal.wav', reconst, fs)

        ppm = (np.abs(dpm) - vft - vift) + mft
        ppm[ppm < 0] = 0
        reconst = ppm * np.angle(dpm) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/smridangam.wav', reconst, fs)

        ppvi = (np.abs(dpvi) - mft - vft) + vift
        ppvi[ppvi < 0] = 0
        reconst = ppvi * np.angle(dpvi) 
        reconst = lb.istft(reconst)
        sf.write(subpath+'/sviolin.wav', reconst, fs)

       




