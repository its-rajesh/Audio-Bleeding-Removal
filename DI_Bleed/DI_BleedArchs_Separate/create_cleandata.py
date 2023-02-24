####################################################################################################
#                         CREATE CLEAN MUSDB FROM BLEEDED MUSDB                                    #
####################################################################################################
# Author: Rajesh R (Git: its-rajesh)
# Date: 24/February/2023

import numpy as np
import os
import librosa as lb
from tensorflow import keras
import DynamicInput as dp
import cmath
import soundfile as sf
from tqdm import tqdm



def calc_istft(mag, phase):
        m, n = mag.shape
        complexx = []
        for i in range(m):
            for j in range(n):
                complexx.append(cmath.rect(mag[i][j], phase[i][j]))

        r = np.array(complexx).reshape(m, n)
        return lb.istft(r)


##############################################################################################################
#                                               DRIVER CODE                                                  #
##############################################################################################################

bleeded_path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/train/"
dest_path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_cleaned/train/"

print("Loading Models")
vocal = keras.models.load_model("/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Vocal/vocal_model.h5")
drums = keras.models.load_model("/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Drums/drums_model.h5")
bass = keras.models.load_model("/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Bass/bass_model.h5")
other = keras.models.load_model("/home/anchal/Desktop/rajesh/DI_Bleed_Archs/Separate/Others/other_model.h5")


bfiles = sorted(os.listdir(bleeded_path))

for i in tqdm(bfiles):
    v, fs = lb.load(bleeded_path+i+'/vocals.wav')
    b, fs = lb.load(bleeded_path+i+'/bass.wav')
    d, fs = lb.load(bleeded_path+i+'/drums.wav')
    o, fs = lb.load(bleeded_path+i+'/other.wav')
    

    stft_v = lb.stft(v)
    stft_b = lb.stft(b)
    stft_d = lb.stft(d)
    stft_o = lb.stft(o)

    inp_v = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_v), 3, 1)
    inp_b = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_b), 3, 1)
    inp_d = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_d), 3, 1)
    inp_o = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft_o), 3, 1)


    predict_v = vocal.predict(inp_v, verbose=1)
    predict_b = bass.predict(inp_b, verbose=1)
    predict_d = drums.predict(inp_d, verbose=1)
    predict_o = other.predict(inp_o, verbose=1)

    predict_v = np.squeeze(predict_v, axis=3)
    predict_b = np.squeeze(predict_b, axis=3)
    predict_d = np.squeeze(predict_d, axis=3)
    predict_o = np.squeeze(predict_o, axis=3)

    decode_v = dp.DynamicInput.decode(dp.DynamicInput, predict_v, 3, 1)
    decode_b = dp.DynamicInput.decode(dp.DynamicInput, predict_b, 3, 1)
    decode_d = dp.DynamicInput.decode(dp.DynamicInput, predict_d, 3, 1)
    decode_o = dp.DynamicInput.decode(dp.DynamicInput, predict_o, 3, 1)

    reconstruct_v = calc_istft(decode_v, np.angle(stft_v))
    reconstruct_b = calc_istft(decode_b, np.angle(stft_b))
    reconstruct_d = calc_istft(decode_d, np.angle(stft_d))
    reconstruct_o = calc_istft(decode_o, np.angle(stft_o))

    out_path = dest_path + i
    os.makedirs(out_path, exist_ok=True)

    sf.write(out_path + '/vocals.wav', reconstruct_v, fs)
    sf.write(out_path + '/bass.wav', reconstruct_b, fs)
    sf.write(out_path + '/drums.wav', reconstruct_d, fs)
    sf.write(out_path + '/other.wav', reconstruct_o, fs)
