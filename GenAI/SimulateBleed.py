from tqdm import tqdm
import h5py
import numpy as np
import os
import pyroomacoustics as pra
import librosa as lb
import random


def RoomImpulseResponse2(source_signals, room_dim, delay_time, mic_pos, source_pos):
    
    vocals, bass, drums = source_signals[0], source_signals[1], source_signals[2]
    fs = 22050
    rt60_tgt = 0.20 #(1.8-2.5 for Orchestra performance, Concert halls are 1.7–2.3 seconds)
    
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Microphone positions [x, y, z] in meters
    #source_pos = np.array([[2, 2, 1], [5, 6, 2], [8, 2, 1]])

    # Sound source positions [x, y, z] in meters
    #mic_pos = np.array([[2, 2.5, 1], [5, 6.1, 1], [8, 2.3, 1]]).T #.T why?
    mic_pos = mic_pos.T

    # Create a ShoeBox room
    #m = pra.Material(energy_absorption="hard_surface")
    #pra.Material(e_absorption)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    #room = pra.ShoeBox(room_dim, fs=fs)
    
    room.add_source(source_pos[0], signal=drums, delay=delay_time)
    room.add_source(source_pos[1], signal=vocals, delay=delay_time)
    room.add_source(source_pos[2], signal=bass, delay=delay_time)
    
    # Add microphone array to the room
    mic_array = pra.MicrophoneArray(mic_pos, fs=fs)
    room.add_microphone_array(mic_array)
    
    room.simulate()
    room.compute_rir()
    
    result = []
    for i, mic in enumerate(mic_array.R.T):
        mic_signal = mic_array.signals[i]
        result.append(mic_signal)
    
    return result

def rescale_positions(original_positions, original_dims, new_dims):
    scaled_positions = []
    for pos in original_positions:
        x, y, z = pos
        x_new = x * new_dims[0] / original_dims[0]
        y_new = y * new_dims[1] / original_dims[1]
        z_new = z * new_dims[2] / original_dims[2]
        scaled_positions.append((x_new, y_new, z_new))
    return scaled_positions

path = '/home/anchal/Desktop/rajesh/research2/Datasets/musdb18hq/train/'
files = sorted(os.listdir(path))

bleeded_songs = []
true_songs = []
print('Simulating Interference...')
for file in tqdm(files):

    if file == '.DS_Store':
        continue

    vocal, fs = lb.load(path+file+'/vocals.wav')
    bass, fs = lb.load(path+file+'/bass.wav')
    drums, fs = lb.load(path+file+'/drums.wav')
    n = len(vocal)
    
    si = np.array([vocal, bass, drums])

    # Original room dimensions
    original_dims = (10, 8, 3)

    # New room dimensions (randomized)
    new_dims = (
        random.uniform(8, 12), 
        random.uniform(6, 10), 
        random.uniform(2.5, 3.5)
    )

    # Original source and microphone positions
    sources = [(2, 2, 1), (5, 6, 2), (8, 2, 1)]
    mics = [(2, 2.5, 1), (5, 6.1, 1), (8, 2.3, 1)]

    # Calculate new positions
    new_sources = rescale_positions(sources, original_dims, new_dims)
    new_mics = rescale_positions(mics, original_dims, new_dims)

    rir_songs = []
    delay_time = np.random.randint(1, 5)*1e-3 #20-50ms
    same_song_outs = RoomImpulseResponse2(si, new_dims, delay_time, mic_pos=np.array(new_mics), source_pos=np.array(new_sources))
    v, b, d = same_song_outs[1][:n], same_song_outs[2][:n], same_song_outs[0][:n]
    rir_songs.append([v, b, d])
    bleeded_songs.append(rir_songs)
    true_songs.append(si)
    
no_of_data = len(bleeded_songs)
print("Writting bleeding data..")
with h5py.File("Xtrain.h5", "w") as f:
    for i in range(no_of_data):
        f.create_dataset("song{}".format(i+1), data=bleeded_songs[i])

        
print("Writting groundtruth data..")
with h5py.File("Ytrain.h5", "w") as f:
    for i in range(no_of_data):
        f.create_dataset("song{}".format(i+1), data=true_songs[i])
        
        
print('Saved Successfully')