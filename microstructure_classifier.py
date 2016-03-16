#%%

import os
import pickle
import sklearn.preprocessing
import numpy
import sklearn.cross_validation
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/autoencoder')

f = open('microstructure.classifier.pickle')
X, Y, index = pickle.load(f)
f.close()

Ns = {}
Ys = {}

def delta(n, i):
    array = numpy.zeros(n)
    array[i] = 1.0

    return array

for i, w in enumerate(index):
    options = list(set([y[i] for y in Y]))

    optionIdxs = dict([(y, x) for x, y in enumerate(options)])

    Ns[w] = len(options)
    Ys[w] = numpy.array([delta(len(options), optionIdxs[y[i]]) for y in Y])

which = ['core', 'type']#index
#%%

#cols = []
#for i in range(len(YN[0])):
#    options = list(set([y[i] for y in YN]))
#
#    optionIdxs = dict([(y, x) for x, y in enumerate(options)])
#
#    cols.append(optionIdxs)

#%%
#options = list(set([y for y in YN]))

#optionIdxs = dict([(y, x) for x, y in enumerate(options)])

#enc = sklearn.preprocessing.OneHotEncoder()
#enc.fit(optionIdxs.values())

#YN2 = [optionIdxs[y] for y in YN]
#Xtrain = X[0:150]
#Yenc = Yenc[0:150]

Xtrain, Xtest, Ytrain0, Ytest0 = sklearn.cross_validation.train_test_split(X, range(len(X)))

N = max(max(Ytrain0), max(Ytest0))

Ytrain = {}
Ytest = {}
for w in index:
    Ytrain[w] = [Ys[w][i] for i in Ytrain0]
    Ytest[w] = [Ys[w][i] for i in Ytest0]
#Ytrain = enc.transform(numpy.array(Ytrain0).reshape(len(Ytrain0), 1)).todense()
#Ytest = enc.transform(numpy.array(Ytest0).reshape(len(Ytest0), 1)).todense()

#%%

import tensorflow as tf
sess = tf.InteractiveSession()

#%%

x = tf.placeholder(tf.float32, shape=[None, 60, 80, 1])

ys = {}
for w in index:
    ys[w] = tf.placeholder(tf.float32, shape=[None, Ns[w]])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#W_conv3 = weight_variable([5, 5, 32, 32])
#b_conv3 = bias_variable([32])

#h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([15 * 20 * 32, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 20 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#
def logistic(D, N, ref):
    W_fc2 = weight_variable([D, N])
    b_fc2 = bias_variable([N])

    y_log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(y_log)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ref, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return { 'loss' : -tf.reduce_sum(ref * tf.log(y_conv)), 'conv' : y_conv, 'acc' : accuracy }

regressions = {}
for w in index:
    regressions[w] = logistic(256, Ns[w], ys[w])

#
#row['core'], row['aged'], row['pct'], row['temp'], row['type']

cross_entropy = None

idxs = []
for w in which:
    dce = regressions[w]['loss']

    if not cross_entropy:
        cross_entropy = dce
    else:
        cross_entropy += dce

cross_entropy2 = None

idxs = []
for w in which + ['temp']:
    dce = regressions[w]['loss']

    if not cross_entropy2:
        cross_entropy2 = dce
    else:
        cross_entropy2 += dce

train_step1 = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)#AdamOptimizer
train_step2 = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy2)#AdamOptimizer

#%%

sess.run(tf.initialize_all_variables())
#%%
process = [train_step2, cross_entropy]

for w in which + ['temp']:
    process.append(regressions[w]['acc'])

for w in which + ['temp']:
    process.append(regressions[w]['loss'])

ces = []
for i in range(1000):
    if i % 10 == 0:
        plt.plot(ces)
        plt.show()

    feed_dict = {
        x: numpy.reshape(Xtrain, (len(Xtrain), Xtrain[0].shape[0], Xtrain[0].shape[1], 1)),
        keep_prob: 0.5
    }

    for w in index:
        feed_dict[ys[w]] = Ytrain[w]

    out = sess.run(process, feed_dict = feed_dict)

    train, ce = out[0:2]
    acc = out[2:2 + len(which + ['temp'])]
    ces2 = out[2 + len(which + ['temp']):2 + 2 * len(which + ['temp'])]

    print 'fwd', zip(which + ['temp'], acc, ces2)

    ces.append(ce)
#%%

process = [cross_entropy]

for w in which + ['temp']:
    process.append(regressions[w]['acc'])

for w in which + ['temp']:
    process.append(regressions[w]['loss'])

feed_dict = {
    x: numpy.reshape(Xtest, (len(Xtest), Xtest[0].shape[0], Xtest[0].shape[1], 1)),
    keep_prob: 1.0
}

for w in index:
    feed_dict[ys[w]] = Ytest[w]

out = sess.run(process, feed_dict = feed_dict)

train = out[0:1]
acc = out[1:1 + len(which + ['temp'])]
ces2 = out[1 + len(which + ['temp']): 1 + 2 * len(which + ['temp'])]

print zip(which + ['temp'], acc, ces2)

#%%
yc2 = [numpy.argmax(y) for y in yc]

print yc2

for y1, yt in zip(yc2, Ytest):
    print options[y1]
    print options[numpy.argmax(yt)]
    print '--'