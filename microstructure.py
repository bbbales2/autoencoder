#%%

import pandas
import os
import skimage.io
import skimage.transform
import numpy
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/lukestuff')

df = pandas.DataFrame.from_csv('scaled_4.csv')

df = df[df['type'] == 'rene']

X_pre = []
Y = []

for idx, row in df.iterrows():
    X_pre.append(skimage.io.imread(row['f']))
    print X_pre[-1].shape
    Y.append(row['core'])
    print row['type']

#%%

X = []

for x in X_pre:
    scaled = skimage.transform.rescale(x, 0.25)
    scaled = (scaled - numpy.mean(scaled.flatten())) / numpy.std(scaled.flatten())

    X.append(scaled)

    #plt.imshow(scaled, interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.show()

#%%

import pickle

f = open('/home/bbales2/autoencoder/microstructure.pickle', 'w')
pickle.dump((X, Y), f)
f.close()

#%%

X = []

for x in X_pre:
    scaled = skimage.transform.rescale(x, 0.30)
    scaled -= scaled.flatten().min()
    scaled /= scaled.flatten().max()
    scaled -= 0.5
    scaled *= 2.0
    #scaled = ((scaled - numpy.mean(scaled.flatten())) / numpy.std(scaled.flatten()))[0:64, 0:64]

    X.append(scaled[0:64, 0:64])

    #plt.imshow(scaled, interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.colorbar()
    #plt.show()

#%%

import pickle

f = open('/home/bbales2/autoencoder/microstructure.64x64.pickle', 'w')
pickle.dump((X, Y), f)
f.close()

