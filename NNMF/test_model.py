import numpy as np
import tensorflow
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape
import museval
import soundfile as sf


def get_metrics(y):
    avg_y = []
    for i in range(len(y)):
        x = y[~np.isnan(y)]
        avg = sum(x)/len(x)
        avg_y.append(avg)
    return avg_y
    

print("Loading Model")
model = tensorflow.keras.models.load_model("model.h5")

X = np.load('X_2mins.npy')
S = np.load('S_2mins.npy')


X = X[:, :, 1323000:1653750] #1-1:15 seconds (Trained..)
S = S[:, :, 1984500:2315250] # 1:30-1:45 seconds (Test..)


n = 4
l = 330750
fs = 22050

print("Prediction...")
predictions = model.predict(X, verbose=1)
predictions = np.squeeze(predictions, axis=3)
print("Predictions Shape:", predictions.shape)

SDR = []
VSDR, BSDR, DSDR, OSDR = [], [], [], []
for i in range(20):
    vsdr, isr, sir, sar = museval.evaluate(predictions[i][0], S[i][0], win=fs, hop=fs)
    bsdr, isr, sir, sar = museval.evaluate(predictions[i][1], S[i][1], win=fs, hop=fs)
    dsdr, isr, sir, sar = museval.evaluate(predictions[i][2], S[i][2], win=fs, hop=fs)
    osdr, isr, sir, sar = museval.evaluate(predictions[i][3], S[i][3], win=fs, hop=fs)
    
    vsdr = get_metrics(vsdr)
    vsdr = sum(vsdr)/len(vsdr)
    print(vsdr)

    
    VSDR.append(vsdr)
    BSDR.append(bsdr)
    DSDR.append(dsdr)
    OSDR.append(osdr)
    SDR.append((vsdr+bsdr+dsdr+osdr)/4)

