import numpy as np
from tqdm import tqdm


path = "/home/rajesh/Projects/DI_Bleed/Dataset/"

data_path = path+'data_second.npy'
label_path = path+'label_second.npy'

def reshape(data):
    out = data[0]
    
    for i in tqdm(range(1, len(data))):
        out = np.concatenate((out, data[i]), axis=0)
    
    return out

print('DATA Loading... (1/2)')
data = np.load(data_path)
print('DATA Loading... (2/2)')
#label = np.load(label_path)

print('DATA Reshaping... (1/2)')
data_res = reshape(data[:20])
print('DATA Reshaping... (2/2)')
#label_res = reshape(label)

print('Writing..')
np.save(path+'test_data.npy', data_res)
#np.save(path+'train_label.npy', label_res)
print(data_res.shape)
print('Completed..!')
