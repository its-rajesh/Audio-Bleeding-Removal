import numpy as np
import solver
from dataloader import MUSDBHQLoader, SaragaLoader
import sys
import argparse
from tqdm import tqdm
import soundfile as sf
import os


parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, default='./Hendry',
                    help="Path which contains sources to remove bleed")
parser.add_argument('--o', type=str, default='./Outputs/',
                    help="Path to store un-bleed audio and sdr")
parser.add_argument('--d', type=str, default='MUSDBHQ',
                    help='What dataset to process (MUSDBHQ/Saraga/Others')
parser.add_argument('--g1', type=int, default=0.001,
                    help='gamma1')
parser.add_argument('--g2', type=int, default=0.5,
                    help='gamma2')
parser.add_argument('--block', type=int, default=25,
                    help='Frames to process')
args = parser.parse_args()

if args.d == 'MUSDBHQ':
    X, m, fs = MUSDBHQLoader.read_file(MUSDBHQLoader, args.i, args.o)
if args.d == 'Saraga':
    X, m, fs = SaragaLoader.read_file(SaragaLoader, args.i, args.o)
print(X.shape, m.shape, fs)
#X = X[:, :44100]
#m = m[:, :44100]
#print(X.shape, m.shape, fs)

block = args.block
gamma1 = args.g1
gamma2 = args.g2
S_pred = []
for i in tqdm(range(0, X.shape[1], block)):
    x_ = X[:, i:i+block]
    m_ = m[:, i:i+block]

    object = solver.minimise
    A_opt, S_opt = solver.minimise.factor(object, X, m, gamma1, gamma2, tol=1e-3, max_iter=1000)
    S_pred.append(S_opt)

S_pred = np.hstack(S_pred)
if args.d == 'MUSDBHQ':
    sf.write(os.path.join(args.o, 'vocal_pred.wav'), S_pred[0], fs)
    sf.write(os.path.join(args.o, 'bass_pred.wav'), S_pred[1], fs)
    sf.write(os.path.join(args.o, 'drums_pred.wav'), S_pred[2], fs)
    sf.write(os.path.join(args.o, 'others_pred.wav'), S_pred[3], fs)
if agrs.d == 'Saraga':
    sf.write(os.path.join(args.o, 'vocal_pred.wav'), S_pred[0], fs)
    sf.write(os.path.join(args.o, 'mridangam_pred.wav'), S_pred[1], fs)
    sf.write(os.path.join(args.o, 'violin_pred.wav'), S_pred[2], fs)

print('Completed.')