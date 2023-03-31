import os
import librosa as lb
import numpy as np

class MUSDBHQLoader:

    def __init__(self):
        return None

    def read_file(self, path, outpath):
        mixture, fs = lb.load(os.path.join(path, 'mixture.wav'))
        vocals, fs = lb.load(os.path.join(path, 'vocals.wav'))
        bass, fs = lb.load(os.path.join(path, 'bass.wav'))
        drums, fs = lb.load(os.path.join(path, 'drums.wav'))
        other, fs = lb.load(os.path.join(path, 'other.wav'))

        if len(mixture) > len(bass):
            n = len(bass)
        else:
            n = len(mixture)

        X = np.array([vocals[:n], bass[:n], drums[:n], other[:n]])
        m = np.array([mixture[:n]])
        assert X.shape[1] == m.shape[1]

        return X, m, fs


class SaragaLoader:

    def __init__(self):
        return None

    def read_file(self):
        mixture, fs = lb.load(os.path.join(path, 'mixture.wav'))
        vocals, fs = lb.load(os.path.join(path, 'vocals.wav'))
        mridangam, fs = lb.load(os.path.join(path, 'mridangam.wav'))
        violin, fs = lb.load(os.path.join(path, 'violin.wav'))

        if len(mixture) > len(mridangam):
            n = len(mridangam)
        else:
            n = len(mixture)

        X = np.array([vocals[:n], mridangam[:n], violin[:n]])
        m = np.array([mixture[:n]])
        assert X.shape[1] == m.shape[1]

        return X, m, fs

