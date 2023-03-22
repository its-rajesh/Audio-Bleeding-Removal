import numpy as np
from scipy.optimize import minimize
import librosa as lb
import soundfile as sf
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

def objective(params, X, m, gamma1, gamma2):
    n, l = X.shape
    A = params[:n**2].reshape((n, n))
    S = params[n**2:].reshape((n, l))
    t = np.sum(S, axis=0)
    return np.linalg.norm(X - np.dot(A, S))**2 + np.linalg.norm(m - t)**2

def update_A(S, X, gamma2):
    n, l = X.shape
    E = np.eye(n) - 1/n
    A = np.dot(X, S.T) + gamma2 * E
    SS = np.dot(S, S.T) + gamma2 * np.eye(l)
    A = np.dot(A, np.linalg.inv(SS))
    np.fill_diagonal(A, 1)
    return A.ravel()

def update_S(A, X):
    return np.linalg.lstsq(A, X, rcond=None)[0].ravel()

def minimize_function(X, m, gamma1, gamma2):

    # initialize A as identity matrix and S as X
    n, l = X.shape
    #print(n, l)
    A_init = np.eye(n)
    S_init = X.copy()

    # set bounds for A and S separately
    A_bounds = [(1, 1) if i == j else (gamma1, gamma2) for i in range(n) for j in range(n)]
    S_bounds = [(-np.inf, np.inf)] * (n * l)
    bounds = A_bounds + S_bounds

    # alternating optimization
    params = np.concatenate([A_init.ravel(), S_init.ravel()])
    res = minimize(objective, params, args=(X, m, gamma1, gamma2), method='Powell', bounds=bounds)

    # extract the optimized A and S
    #print(n, l)
    A_opt = res.x[:n**2].reshape((n, n))
    S_opt = res.x[n**2:].reshape((n, l))
    
    return A_opt, S_opt


print('Reading Input Audio...')
bleed_path = '/home/rajesh/Desktop/Datasets/musdb18hq_bleeded/train/Music Delta - Hendrix/'
bvocals, fs = lb.load(bleed_path+'vocals.wav')
bbass, fs = lb.load(bleed_path+'bass.wav')
bdrums, fs = lb.load(bleed_path+'drums.wav')
bother, fs = lb.load(bleed_path+'other.wav')

clean_path = '/home/rajesh/Desktop/Datasets/musdb18hq/train/Music Delta - Hendrix/'
vocals, fs = lb.load(clean_path+'vocals.wav')
bass, fs = lb.load(clean_path+'bass.wav')
drums, fs = lb.load(clean_path+'drums.wav')
other, fs = lb.load(clean_path+'other.wav')

mixture, fs = lb.load(clean_path+'mixture.wav')

if len(bbass) > len(bass):
    n_ = len(bass)
else:
    n_ = len(bbass)
    
if n_ > len(mixture):
    n_ = len(mixture)
    
vocals = vocals[:n_]
bass = bass[:n_]
drums = drums[:n_]
other = other[:n_]

bvocals = bvocals[:n_]
bbass = bbass[:n_]
bdrums = bdrums[:n_]
bother = bother[:n_]

m = mixture[:n_]

X = np.array([bvocals, bbass, bdrums, bother])
S = np.array([vocals, bass, drums, other])

X = X[:, :88200]
S = S[:, :88200]
m = m[:88200]

print(X.shape, S.shape, m.shape)

    
blocks = 25
print('Dividing {} blocks...'.format(blocks))

n_, l_ = X.shape
XChunk, SChunk, mChunk = [], [], []
for i in range(0, l_, blocks):
    XChunk.append(X[:, i:i+blocks])
    SChunk.append(X[:, i:i+blocks])
    mChunk.append(X[:, i:i+blocks])

run = len(mChunk)

gamma1 = 0
gamma2 = 0.9

S_arr = []
print('Optimizing...')
for i in range(run):
    A_opt, S_opt = minimize_function(XChunk[i], mChunk[i], gamma1, gamma2)
    l1 = np.linalg.norm(XChunk[i] - np.dot(A_opt, S_opt))
    l2 = np.linalg.norm(mChunk[i] - (S_opt[0]+S_opt[1]+S_opt[2]+S_opt[3]))
    print("[BLOCK: {}/{}] ||X-AS||^2: {}, ||m-t||^2: {}, LOSS: {}".format(i+1, run, np.round(l1, 5), np.round(l2, 5), np.round(l1+l2, 5)))
    S_arr.append(S_opt)

S_pred = np.hstack(S_arr)

print('Writing Files...')
out = '/home/rajesh/Desktop/'
sf.write(out+'pred_vocal.wav', S_pred[0], fs)
sf.write(out+'pred_bass.wav', S_pred[1], fs)
sf.write(out+'pred_drums.wav', S_pred[2], fs)
sf.write(out+'pred_other.wav', S_pred[3], fs)
print('DONE!!!')