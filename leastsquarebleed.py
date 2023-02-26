import os
import librosa as lb
import soundfile as sf


####################################################################################################
# LOADING BLEED DATA

bleed_path = '/home/rajesh/Desktop/Datasets/musdb18hq_bleeded/train/A Classic Education - NightOwl/'
bvocals, fs = lb.load(bleed_path+'vocals.wav')
bbass, fs = lb.load(bleed_path+'bass.wav')
bdrums, fs = lb.load(bleed_path+'drums.wav')
bother, fs = lb.load(bleed_path+'other.wav')

R = np.array([bvocals, bbass, bdrums, bother])
print(R.shape)


####################################################################################################
# LOADING TRUE SOURCES

clean_path = '/home/rajesh/Desktop/Datasets/musdb18hq/train/A Classic Education - NightOwl/'
vocals, fs = lb.load(clean_path+'vocals.wav')
bass, fs = lb.load(clean_path+'bass.wav')
drums, fs = lb.load(clean_path+'drums.wav')
other, fs = lb.load(clean_path+'other.wav')

S = np.array([vocals[:3780252], bass[:3780252], drums[:3780252], other[:3780252]])
print(S.shape)

####################################################################################################
# LEAST SQUARE SOLUTION

def leastnormsoln(r, s):
    b = np.array([r-s]).T
    A = np.array([[s[1], s[2], s[3], 0, 0 ,0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, s[0], s[2], s[3], 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, s[0], s[1], s[3], 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, s[0], s[1], s[2]]])
    soln, residual, rank, singular = linalg.lstsq(A, b, rcond=-1)
    return np.array([[1, soln[0][0], soln[1][0], soln[2][0]],
                     [soln[3][0], 1, soln[4][0], soln[5][0]],
                     [soln[6][0], soln[7][0], 1, soln[8][0]],
                     [soln[9][0], soln[10][0], soln[11][0], 1]])


lmda = []
for i in range(R.shape[1]):
    r = R[:, i]
    s = S[:, i]
    lmda.append(leastnormsoln(r, s))


####################################################################################################
# GETTING BACK TRUE SOURCES

def predict(r, l):
    lambda_inv = linalg.inv(l)
    soln = np.dot(lambda_inv,r)
    return soln

coln_soln = []
for i in range(R.shape[1]):
    r = R[:, i]
    l = lmda[i]
    coln_soln.append(predict(r, l))

unbleeded = np.array(coln_soln).T

unbleed_vocal = unbleeded[0]
unbleed_bass = unbleeded[1]
unbleed_drum = unbleeded[2]
unbleed_other = unbleeded[3]

sf.write('/home/rajesh/Desktop/vocal.wav', unbleed_vocal, fs)
sf.write('/home/rajesh/Desktop/bass.wav', unbleed_bass, fs)
sf.write('/home/rajesh/Desktop/drum.wav', unbleed_drum, fs)
sf.write('/home/rajesh/Desktop/other.wav', unbleed_other, fs)