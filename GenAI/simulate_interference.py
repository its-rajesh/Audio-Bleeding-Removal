'''
LATEST FUNCTION TO SIMULATE INTERFERENCE
- LARGE ROOMS, LARGE RIR
- FIXED ROOM DIMENSIONS
- CORRECT BLEED
- IGNORING SILENT AUDIOS (CAUSING NEGATIVE SDRS)
'''

import librosa as lb
import numpy as np
from tqdm import tqdm
import pyroomacoustics as pra


def detect_activity(y, min_duration=2.0, chunk_size=0.1, threshold_db=-40):
    """
    Detect if an audio file has at least `min_duration` seconds of continuous activity.
    
    Args:
        audio_path: Path to the audio file.
        min_duration: Minimum required duration (seconds) of continuous activity.
        chunk_size: Window size (seconds) for RMS calculation.
        threshold_db: Threshold in dB below which audio is considered silent.
    
    Returns:
        bool: True if audio has ≥2s of activity, else False.
    """
    sr = 22050
    
    # Calculate RMS energy in chunks
    chunk_samples = int(chunk_size * sr)
    rms = []
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        if len(chunk) == 0:
            break
        rms_val = np.sqrt(np.mean(chunk**2))
        rms.append(rms_val)
    
    # Convert RMS to dB
    rms_db = 10 * np.log10(np.array(rms) + 1e-10)  # Avoid log(0)
    
    # Thresholding: Active if RMS > threshold_db
    active = rms_db > threshold_db
    
    # Find continuous active segments
    active_segments = []
    current_segment = 0.0
    for is_active in active:
        if is_active:
            current_segment += chunk_size
        else:
            if current_segment >= min_duration:
                return True
            current_segment = 0.0
    # Check final segment
    if current_segment >= min_duration:
        return True
    
    return False
    
def RoomImpulseResponse(source_signals, room_dim, delay_time):
    
    vocals, bass, drums = source_signals[0], source_signals[1], source_signals[2]
    fs = 22050
    rt60_tgt = 0.8 #seconds reverberation
    
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Microphone positions [x, y, z] in meters
    source_pos = np.array([[2, 33, 2], [20, 17, 18], [38, 2, 2]])

    # Sound source positions [x, y, z] in meters
    mic_pos = np.array([[2, 30, 2], [20, 16, 18], [36, 2, 2]]).T

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
        result.append(mic_signal[:len(vocals)])
    
    return result


def remove_zero_sources(s):
    results = [detect_activity(i, min_duration=2.0, chunk_size=0.1, threshold_db=-40) for i in si]
    if all(results):
        return True
    else:
        return False



path = "/home/anchal/Desktop/rajesh/Clean/Dataset/MUSDBHQ/LM/Ytrain.npy"

Ytrain = np.load(path)
print(Ytrain.shape)
Ytrain = Ytrain[:, :3, :]
print(Ytrain.shape)

rir_songs = []
true_songs = []
for si in tqdm(Ytrain[:, :, :]):
    # Room dimensions [length, width, height] in meters
    #room_dim = [np.random.randint(10, 15), np.random.randint(10, 15), np.random.randint(8, 12)]
    room_dim = [40, 35, 20]
    #delay_time = np.random.randint(1, 5)*1e-3 #20-50ms
    delay_time = 0 #1e-6
    if remove_zero_sources(si):
        same_song_outs = RoomImpulseResponse(si, room_dim, delay_time)
        ve, be, de = same_song_outs[1], same_song_outs[2], same_song_outs[0]
        rir_songs.append([ve, be, de])
        true_songs.append(si)
    else:
        print('Less activity detected, Ignored!')
        
        
np.save('Xtrain.npy', rir_songs)
np.save('Ytrain.npy', true_songs)

print("DONE")

