import numpy as np
import os
import librosa as lb
from tensorflow import keras
import DynamicInput as dp
import cmath
from matplotlib import pyplot as plt
import museval
import soundfile as sf
import librosa.display



bleed_path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq_bleeded/test/The Long Wait - Dark Horses/"
label_path = "/home/anchal/Desktop/rajesh/Datasets/musdb18hq/test/The Long Wait - Dark Horses/"

print("Loading Model")
di3_dca = keras.models.load_model("/home/anchal/Desktop/rajesh/complex_di/complexnn.h5")

print("Loading Files")
xtest, fs = lb.load(bleed_path+'vocals.wav', sr=22050)
ytest, fs = lb.load(label_path+'vocals.wav', sr=22020)

print(xtest.shape, ytest.shape)
if len(xtest) > len(ytest):
    xtest = xtest[:len(ytest)]
else:
    ytest = ytest[:len(xtest)] 
print(xtest.shape, ytest.shape)

stft = lb.stft(xtest)
inp = dp.DynamicInput.encode3D(np.abs(stft), 3, 1)

print("Evaluation...")
y_stft = lb.stft(ytest)
ytest_inp = dp.DynamicInput.encode3D(np.abs(y_stft), 3, 1)
loss = di3_dca.evaluate(inp,  ytest_inp, verbose=1)
print("Test Loss: {}".format(loss))


print("Prediction...")
predictions = di3_dca.predict(inp, verbose=1)
predictions = np.squeeze(predictions, axis=4)
print("Predictions Shape:", predictions.shape)

decoded = dp.DynamicInput.decode3D(predictions, 3, 1)
decoded_y = dp.DynamicInput.decode3D(ytest_inp, 3, 1)

print("DECODED:", decoded.shape, decoded_y.shape)
print("ACTUAL:", stft.shape, y_stft.shape)

def calc_istft(stft):
    mag = stft[0]
    phase = stft[1]
    m, n = mag.shape
    complex_ = []
    for i in range(m):
        for j in range(n):
            complex_.append(complex(mag[i][j], phase[i][j]))
    
    complex_ = np.array(complex_).reshape(m, n)
    
    return lb.istft(complex_)


##############################################################################################################

print("Working on ISTFT.....")
reconstructed = calc_istft(decoded)
true = calc_istft(decoded_y)

print(true.shape, reconstructed.shape)


print("Writting Audio Files!")
out_path = os.getcwd() + '/Outputs'
os.makedirs(out_path, exist_ok=True)

sf.write(out_path + '/reconstructed.wav', reconstructed, fs)
sf.write(out_path + '/true.wav', true, fs)

##############################################################################################################

print("Calculating Evaluation Metrics")
print("fs:", fs)
t = np.array([true])
r = np.array([reconstructed])

print(t.shape, r.shape)
sdr, isr, sir, sar = museval.evaluate(t, r, win=fs, hop=fs)


def get_metrics(y):
    avg_y = []
    for i in range(len(y)):
        x = y[~np.isnan(y)]
        avg = sum(x)/len(x)
        avg_y.append(avg)
    return avg_y
    
    
avg_sdr = get_metrics(sdr)
avg_isr = get_metrics(isr) #Source to Spatial Distortion Image
avg_sir = get_metrics(sir)
avg_sar = get_metrics(sar)

print('Overall SDR: ', sum(avg_sdr)/len(avg_sdr))
print('Overall ISR: ', sum(avg_isr)/len(avg_isr))
print('Overall SIR: ', sum(avg_sir)/len(avg_sir))
print('Overall SAR: ', sum(avg_sar)/len(avg_sar))

##############################################################################################################

fig, ax = plt.subplots()
img = lb.display.specshow(decoded[0], x_axis='time', y_axis='log', ax=ax)
ax.set(title='Bleed Removed Spect')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/bleed_removed.svg')


fig, ax = plt.subplots()
img = lb.display.specshow(decoded_y[0], x_axis='time', y_axis='log', ax=ax)
ax.set(title='Reconst Original Spect')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/reconst_original.svg')

fig, ax = plt.subplots()
img = lb.display.specshow(np.abs(y_stft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Original Spect')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/original.svg')

fig, ax = plt.subplots()
img = lb.display.specshow(np.abs(stft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Bleeded Spect')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/bleeded.svg')

print("END")
