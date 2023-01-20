import os
import librosa as lb
import numpy as np
import re
from tqdm import tqdm
import soundfile as sf


import DynamicInput as dp


Bpath = "/home/rajesh/Projects/DI_Bleed/Dataset/AIbleed/"
Gpath = "/home/rajesh/Projects/DI_Bleed/Dataset/Actual/"

files = os.listdir(Bpath)

data, label = [], []
for i in tqdm(files[50:]): #NOTE HERE IMPORTANT!!!!
    #print(i)
    name = re.split('(\d+)', i)
    #print(name)
    gname = name[0]+name[1]+name[-1]
    #print(gname)
    d, fs = sf.read(Bpath+i)
    l, fs = sf.read(Gpath+gname)
    
    d_stft = lb.stft(d)
    l_stft = lb.stft(l)
    
    data.append(dp.DynamicInput.encode(dp.DynamicInput, d_stft, 3, 1))
    label.append(dp.DynamicInput.encode(dp.DynamicInput, l_stft, 3, 1))


data = np.array(data)
label = np.array(label)

out_path = "/home/rajesh/Projects/DI_Bleed/Dataset/"

np.save(out_path+'data.npy', data)
np.save(out_path+'label.npy', label)