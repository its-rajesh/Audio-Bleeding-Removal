import numpy as np
import h5py
import librosa as lb
from tqdm import tqdm


#### DATASET PREPROCESSING

def get_stft(chunk):
    stft = []
    for i in range(3):
        stft.append(lb.stft(chunk[i], n_fft=2047))
    return np.array(stft)

def process_h5py_to_fixed_length_chunks(h5py_filename, chunk_length=22050, flag=True):
    """
    Reads audio arrays from an HDF5 file (stored as (1, 3, l)),
    splits them into fixed-length chunks (3, chunk_length),
    and discards incomplete chunks. Combines the chunks into a single array.

    Parameters:
        h5py_filename (str): Path to the HDF5 file containing audio arrays.
        chunk_length (int): Desired chunk length for each chunk (default is 22050).
        flag (boolean): True for Xtrain, False for Ytrain

    Returns:
        np.ndarray: Array of shape (n, 3, chunk_length) containing all chunks.
    """
    # List to hold all valid chunks
    all_chunks = []

    # Open the HDF5 file
    with h5py.File(h5py_filename, "r") as f:
        # Iterate through all datasets in the file
        for key in tqdm(f.keys()):
            # Load the current audio array and squeeze dimensions
            if flag:
                audio = f[key][0]  # Shape becomes (3, l) after squeezing
            else:
                audio = f[key]
            
            # Get the number of samples (l)
            num_samples = audio.shape[1]
            
            # Process the audio into chunks of shape (3, chunk_length)
            for start in range(0, num_samples, chunk_length):
                chunk = audio[:, start:start + chunk_length]
                
                # Only keep chunks of the exact desired shape
                if chunk.shape[1] == chunk_length:
                    stft = get_stft(chunk)
                    all_chunks.append(stft)
    
    # Convert the list of chunks to a single NumPy array
    all_chunks_array = np.array(all_chunks)  # Shape will be (n, 3, chunk_length)

    return all_chunks_array


print('Preprocessing data...')
Ytrain = process_h5py_to_fixed_length_chunks("/home/anchal/Desktop/rajesh/research2/cGANIR/Ytrain.h5", flag=False)
Xtrain = process_h5py_to_fixed_length_chunks("/home/anchal/Desktop/rajesh/research2/cGANIR/Xtrain.h5", flag = True)
print('Shapes:', Xtrain.shape, Ytrain.shape)


np.save('Xtrain_stft.npy', Xtrain)
np.save('Ytrain_stft.npy', Ytrain)

print('Done')