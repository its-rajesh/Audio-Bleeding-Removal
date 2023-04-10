import numpy as np
from tqdm import tqdm


print('Loading Files...')
path = "/home/anchal/Desktop/rajesh/3Source_TDCr/"


x1 = np.load('xtrain_0_10.npy')
x2 = np.load('xtrain_10_30.npy')
x3 = np.load('xtrain_30_50.npy')
x4 = np.load('xtrain_50_80.npy')
x5 = np.load('xtrain_80_100.npy')

y1 = np.load('ytrain_0_10.npy')
y2 = np.load('ytrain_10_30.npy')
y3 = np.load('ytrain_30_50.npy')
y4 = np.load('ytrain_50_80.npy')
y5 = np.load('ytrain_80_100.npy')

x6 = np.load('xtest_0_25.npy')
x7 = np.load('xtest_25_50.npy')

y6 = np.load('ytest_0_25.npy')
y7 = np.load('ytest_25_50.npy')

xtrain = np.vstack([x1, x2, x3, x4, x5])
ytrain = np.vstack([y1, y2, y3, y4, y5])

xtest = np.vstack([x6, x7])
ytest = np.vstack([y6, y7])

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

np.save('xtrain.npy', xtrain)
np.save('ytrain.npy', ytrain)
np.save('xtest.npy', xtest)
np.save('ytest.npy', ytest)