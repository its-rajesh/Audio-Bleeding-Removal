import librosa as lb
import numpy as np
import os
import model
import cmath
import soundfile as sf

##############################################################################################################

'''
Change the appropriate paths here. Training and its label path. Similarly test and its label path.
'''

# Training data path
trpath = '/Users/rajeshr/Desktop/MUSDB18BLEEDTEST/training/'
# Training Ground Truth path
labelpath = '/Users/rajeshr/Desktop/MUSDB18BLEEDTEST/trainingLabel/'
# Test data path
testpath = '/Users/rajeshr/Desktop/MUSDB18BLEEDTEST/test/'
# Test Ground Truth path
testlabelpath = '/Users/rajeshr/Desktop/MUSDB18BLEEDTEST/testLabel/'

##############################################################################################################

'''
FUNCTION: Calculating STFT (SR=8Khz, n_fft=512) 
NOTE: All files will be limited to 2 minutes of duration! (2*60*8000=960000 samples)
Inputs: path with audio files
Returns: numpy array of STFT
'''


def calc_stft(path):
    stft = []
    for file in os.listdir(path):
        try:
            f, sr = lb.load(path + file, sr=8000, mono=True)
            stft.append(np.abs(lb.stft(f[:960000], n_fft=512)))
        except:
            print('Some ERROR occured while reading!')
    return np.array(stft)


'''
FUNCTION: Calculating PHASE (SR=8Khz, n_fft=512) 
NOTE: All files will be limited to 2 minutes of duration! (2*60*8000=960000 samples)
Inputs: path with audio files
Returns: numpy array of STFT
'''


def get_phase(path):
    phase = []
    for file in os.listdir(path):
        try:
            f, sr = lb.load(path + file, sr=8000, mono=True)
            phase.append(np.angle(lb.stft(f[:960000], n_fft=512)))
        except:
            print('Some ERROR occured while reading!')
    return np.array(phase)


##############################################################################################################

print('Calculating STFT for Training Data')
training_stft = calc_stft(trpath)
print('Calculating STFT for GroundTruth Training Data')
training_label_stft = calc_stft(labelpath)

print('Calculating STFT for Test Data')
test_stft = calc_stft(testpath)
print('Calculating STFT for GroundTruth Test Data')
test_label_stft = calc_stft(testlabelpath)

print('Calculating Phase for Training Data')
test_label_phase = get_phase(trpath)  # calculating phase for input data

##############################################################################################################

x_train = training_stft
y_train = training_label_stft
X_TEST = test_stft
Y_TEST = test_label_stft

print('Data Processing Done..')
obj = model.RemoveBleed(x_train, y_train, X_TEST, Y_TEST)

predictions = obj.predictions
savemodelpath = './models/bleed_model.h5'
obj.bleed_removal.save(savemodelpath)
print('FINAL MODEL SAVED SUCCESSFULLY IN ({})'.format(savemodelpath))


##############################################################################################################

def convert_complex(mag, phase):
    m, n = mag.shape
    complexx = []
    for i in range(m):
        for j in range(n):
            complexx.append(cmath.rect(mag[i][j], phase[i][j]))
    return np.array(complexx).reshape(m, n)


def calc_istft(mag, phase):
    signal = []
    for i in range(len(mag)):
        D = convert_complex(mag[i], phase[i])
        signal.append(lb.istft(D))

    return np.array(signal)


##############################################################################################################

print("Working on ISTFT.....")
reconstructed = calc_istft(predictions, test_label_phase)

##############################################################################################################

print("Writting Audio Files!")
out_path = os.getcwd() + '/Outputs'
os.makedirs(out_path, exist_ok=True)
k = 1
n = len(reconstructed)
for audio in reconstructed:
    sf.write(out_path + '/reconstructed{}.wav'.format(k), audio, 8000)
    k += 1
    if k % 10 == 0:
        print("{} files written out of {}".format(k, n))
print("END")

##############################################################################################################

