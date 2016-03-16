#%%

import pandas
import os
import skimage.io
import skimage.transform
import numpy
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/lukestuff')

df = pandas.DataFrame.from_csv('scaled_2.csv')

#df = df[df['type'] == 'rene']

X_pre = []
Y = []

for idx, row in df.iterrows():
    X_pre.append(skimage.io.imread(row['f']))
    print X_pre[-1].shape
    Y.append((row['core'], row['aged'], row['pct'], row['temp'], row['type']))
    print row['core']
    print row['type']

X = []

for x in X_pre:
    scaled = skimage.transform.resize(x, (60, 80))
    scaled = (scaled - numpy.mean(scaled.flatten())) / numpy.std(scaled.flatten())

    X.append(scaled)

    #plt.imshow(scaled, interpolation = 'NONE', cmap = plt.cm.gray)
    #plt.show()

import pickle

f = open('/home/bbales2/autoencoder/microstructure.2.classifier.pickle', 'w')
pickle.dump((X, Y, numpy.array(['core', 'aged', 'pct', 'temp', 'type'])), f)
f.close()
