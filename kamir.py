import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import medfilt2d

#-----------------------------------------------------
# KAMIR
# Kernel Additive Models for Interference Reduction
# Python conversion of Antoine Liutkus (Inria, 2014)
#-----------------------------------------------------


# Parameters
datadir_template = '/Users/rajeshr/Desktop/outpus/kamir/{}/'
outdir_template = '/Users/rajeshr/Desktop/outpus/kamir/{}/'
my_list = ['ex1', 'ex2', 'ex3']
nfft = 8192                    # length of STFT window
overlap = 0.9                  # overlap ratio
hop_length = int(nfft * (1 - overlap))
Lmax = 500                     # max length in seconds
minleakage = 0.05              # minimum leakage
selection_threshold = 0.8      # threshold on L
alpha = 2.0                    # alpha (2=gaussian)
approx = False                # use logistic approx?
slope = 20.0
tresh = 0.6
niter = 5                      # number of KAMIR iterations
good_init = True              # user-defined init
learn_L = True                 # learn L?
beta = 0.0                     # beta divergence
proximityKernel = 1            # median-filter kernel (scalar or 2D array)



def kamir_batch():
    for example_id in my_list:
        datadir = datadir_template.format(example_id)
        outdir = outdir_template.format(example_id)
        os.makedirs(outdir, exist_ok=True)

        # list .wav files
        wavs = sorted(f for f in os.listdir(datadir) if f.lower().endswith('.wav'))
        if not wavs:
            print(f'No files in {datadir}, skipping.')
            continue

        # load and truncate signals
        sigs, fs = [], None
        for fname in wavs:
            path = os.path.join(datadir, fname)
            data, sr = sf.read(path)
            fs = sr if fs is None else fs
            assert sr == fs, 'Sample rates differ.'
            if data.ndim > 1:
                data = data[:, 0]
            data = data[:int(Lmax * fs)]
            sigs.append(data)

        I = 3#len(sigs)
        N = min(len(s) for s in sigs)
        sigs = [s[:N] for s in sigs]

        # STFT
        X = np.stack([
            librosa.stft(s, n_fft=nfft, hop_length=hop_length,
                         win_length=nfft, window='hann')
            for s in sigs
        ], axis=2).astype(np.complex64)
        F, T, _ = X.shape

        # dynamic number of sources equals channels
        J = I
        sources_names = [f'source{j+1}' for j in range(J)]

        # initialize L
        L0 = np.eye(J, dtype=float)
        if good_init:
            Lmat = np.maximum(L0, selection_threshold)
            niter_L = 1
        else:
            step = I / J
            Lmat = np.zeros((I, J), dtype=float)
            for j in range(J):
                start = int(round(j * step))
                end = int(round((j + 1) * step))
                Lmat[start:end, j] = 1
            Lmat = np.maximum(Lmat, selection_threshold)
            niter_L = 3
        # expand to (F, I, J)
        L = np.tile(Lmat[np.newaxis, :, :], (F, 1, 1))

        # gain equalization
        print('Identifying channel gains...')
        V = np.abs(X) ** alpha
        gains = np.quantile(V, 0.05, axis=1) ** (1 / alpha)  # (F, I)
        print('Compensating gains...')
        X = X * (1.0 / gains)[:, np.newaxis, :]

        # initialize P
        P = np.zeros((F, T, J), dtype=float)
        for j in range(J):
            chs = np.where(L0[:, j] > 0)[0]
            P[:, :, j] = np.mean(V[:, :, chs], axis=2)
            if isinstance(proximityKernel, np.ndarray) and proximityKernel.size > 1:
                P[:, :, j] = medfilt2d(P[:, :, j], kernel_size=proximityKernel.shape)

        # iterative BackFitting
        for it in range(niter + 1):
            print(f'KAMIR iter {it+1}/{niter+1} on example {example_id}')
            for j in range(J):
                close_ch = np.where(L0[:, j] > 0)[0]
                # compute Y for all close channels
                Y = np.stack([
                    ((L[:, ch, j][:, None] * P[:, :, j]) /
                     (np.finfo(float).eps + sum(L[:, ch, j2][:, None] * P[:, :, j2]
                         for j2 in range(J)))) * X[:, :, ch]
                    for ch in close_ch
                ], axis=2)

                # final iteration: reconstruct and save
                if it == niter:
                    recon_mult = np.zeros((N, len(close_ch)), dtype=float)
                    for ni, ch in enumerate(close_ch):
                        Yg = Y[:, :, ni] * gains[:, ch][:, None]
                        recon = librosa.istft(Yg, hop_length=hop_length,
                                              win_length=nfft, window='hann', length=N)
                        recon_mult[:, ni] = recon
                    outpath = os.path.join(outdir, f'pred_{sources_names[j]}.wav')
                    sf.write(outpath, recon_mult, fs)
                    continue

                # update P for next iteration
                invL = 1.0 / L[:, close_ch, j]  # (F, nclose)
                tmp = (np.abs(Y) ** alpha) * invL[:, None, :]
                P[:, :, j] = np.mean(tmp, axis=2)
                if isinstance(proximityKernel, np.ndarray) and proximityKernel.size > 1:
                    P[:, :, j] = medfilt2d(P[:, :, j], kernel_size=proximityKernel.shape)

            # update L
            if learn_L and it < niter:
                for _ in range(niter_L):
                    for i_ch in range(I):
                        denom = np.finfo(float).eps + sum(
                            L[:, i_ch, j2][:, None] * P[:, :, j2] for j2 in range(J)
                        )
                        for j in range(J):
                            if not np.isnan(beta):
                                num = np.sum(denom**(beta-2) * V[:, :, i_ch] * P[:, :, j], axis=1)
                                den = np.sum(denom**(beta-1) * np.maximum(P[:, :, j], np.finfo(float).eps), axis=1)
                            else:
                                dist = np.abs(denom - V[:, :, i_ch] + np.finfo(float).eps)**(alpha-2)
                                num = np.sum(V[:, :, i_ch] * dist * P[:, :, j], axis=1)
                                den = np.sum(denom * dist * P[:, :, j], axis=1)
                            L[:, i_ch, j] *= num / den
                    Lsum = np.sum(L, axis=2)
                    L /= Lsum[:, :, None]
                    L = np.maximum(L, minleakage)

        print(f'Finished example {example_id}')
