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



path = "/home/s21005/research/di_bleed/3g_dca/test/"

print("Loading Model")
di3_dca = keras.models.load_model("/home/s21005/research/di_bleed/3g_dca/bleed_CAE.h5")

print("Loading Files")
xtest, fs = lb.load(path+'vocal2bass8.wav')
ytest, fs = lb.load(path+'vocal2.wav')

print(fs)
stft = lb.stft(xtest)
inp = dp.DynamicInput.encode(dp.DynamicInput, np.abs(stft), 3, 1)

print("Evaluation...")
y_stft = lb.stft(ytest)
ytest_inp = dp.DynamicInput.encode(dp.DynamicInput, np.abs(y_stft), 3, 1)
loss = di3_dca.evaluate(inp,  ytest_inp, verbose=1)
print("Test Loss: {}".format(loss))


print("Prediction...")
predictions = di3_dca.predict(inp, verbose=1)
predictions = np.squeeze(predictions, axis=3)
print("Predictions Shape:", predictions.shape)

decoded = dp.DynamicInput.decode(dp.DynamicInput, predictions, 3, 1)
decoded_y = dp.DynamicInput.decode(dp.DynamicInput, ytest_inp, 3, 1)

print("DECODED:", decoded.shape, decoded_y.shape)
print("ACTUAL:", stft.shape, y_stft.shape)

def calc_istft(mag, phase):
    m, n = mag.shape
    complexx = []
    for i in range(m):
        for j in range(n):
            complexx.append(cmath.rect(mag[i][j], phase[i][j]))

    r = np.array(complexx).reshape(m, n)
    return lb.istft(r)


##############################################################################################################

print("Working on ISTFT.....")
reconstructed = calc_istft(decoded, np.angle(stft))
true = calc_istft(decoded_y, np.angle(y_stft))

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
img = lb.display.specshow(decoded, x_axis='time', y_axis='log', ax=ax)
ax.set(title='Reconstructed')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/reconst.svg')


fig, ax = plt.subplots()
img = lb.display.specshow(decoded_y, x_axis='time', y_axis='log', ax=ax)
ax.set(title='True')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/true.svg')

fig, ax = plt.subplots()
img = lb.display.specshow(np.abs(y_stft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Original')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/original.svg')

fig, ax = plt.subplots()
img = lb.display.specshow(np.abs(stft), x_axis='time', y_axis='log', ax=ax)
ax.set(title='Bleeded')
fig.colorbar(img, ax=ax, format="%+2.f dB")
fig.savefig(out_path+'/bleeded.svg')

print("END")
