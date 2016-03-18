#%%

import matplotlib.pyplot as plt

import os
import pickle
import sklearn.preprocessing
import numpy
import sklearn.cross_validation

os.chdir('/home/bbales2/autoencoder')

f = open('microstructure.64x64.pickle')
X, Y = pickle.load(f)
f.close()

XY = zip(X, Y)

numpy.random.shuffle(XY)
X, Y = zip(*XY)

enc = sklearn.preprocessing.OneHotEncoder()
enc.fit([[True], [False]])

#Xtrain = X[0:150]
#Yenc = Yenc[0:150]

Xtrain, Xtest, Ytrain0, Ytest0 = sklearn.cross_validation.train_test_split(X, Y)

Ytrain = enc.transform(numpy.array(Ytrain0).reshape(len(Ytrain0), 1)).todense()
Ytest = enc.transform(numpy.array(Ytest0).reshape(len(Ytest0), 1)).todense()

#%%

import tensorflow as tf
sess = tf.InteractiveSession()

#%%

x = tf.placeholder(tf.float32, shape=[None, 4096])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

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

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(x, [-1, 64, 64, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([16 * 16 * 32, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 2])
b_fc2 = bias_variable([2])

y_log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(y_log)

yloss = -tf.reduce_sum(y_*tf.log(y_conv))

cross_entropy = yloss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

firstNetworkVars = set(tf.all_variables())

#%%

sess.run(tf.initialize_variables(firstNetworkVars))

batch_size = len(Xtrain) - 1
ces = []
for i in range(20000):
    start = 0#(i * batch_size) % len(Xtrain)
    stop = len(Xtrain)#((i + 1) * batch_size) % len(Xtrain)

    if start > stop:
        continue

    batch = [im.flatten() for im in Xtrain[start : stop]]
    batch_y = Ytrain[start : stop] * 1

    if i % 10 == 0:
        plt.plot(ces)
        plt.show()

    train, acc, ce = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: batch, y_: batch_y, keep_prob: 0.5})
    print 'fwd', acc, ce

    ces.append(ce)

    #train = sess.run([train_deconv_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: batch, y_: batch_y, keep_prob: 1.0}))#
#%%
hfc1 = sess.run([h_fc1], feed_dict={x: [batch[0]], keep_prob: 1.0})[0]
#%%

#%%

dt = 0.01
dx = 1

w = 2.0 * numpy.pi * numpy.fft.fftfreq(64, 1.0)
w2 = w * w
wx2, wy2 = numpy.meshgrid(w2, w2)

alpha = 1.0
y = 0.25

dximage = weight_variable([1, 64, 64, 1])

zeros = tf.zeros([1, 64, 64, 1])

#dximage = tf.add(zeros, signal.reshape([1, 64, 64, 1]))
Floss = tf.reduce_sum(tf.sub(tf.mul(tf.mul(tf.mul(dximage, dximage), dximage), dximage), tf.mul(dximage, dximage)))
Dtmp = tf.fft2d(tf.reshape(tf.complex(dximage, zeros), [64, 64]))
Ddiff = tf.real(tf.ifft2d(tf.mul((y * (wx2 + wy2)).astype('complex64'), Dtmp)))
Dloss = tf.nn.l2_loss(Ddiff)

fftc = tf.reshape(tf.complex(tf.mul(tf.mul(dximage, dximage), dximage), zeros), [64, 64])#tf.fft2d()
dximage2 = tf.fft2d(tf.reshape(tf.complex(dximage, zeros), [64, 64]))#tf.complex(tf.mul(tf.mul(dximage, dximage), dximage), zeros)
dximage3 = tf.div(dximage2, (1 + alpha * (wx2 + wy2) * dt * (-1 + y * (wx2 + wy2))).astype('complex64'))
dximage4 = tf.sub(dximage3, tf.mul((alpha * (wx2 + wy2) * dt).astype('complex64'), fftc))

rdximage = tf.reshape(tf.real(tf.ifft2d(dximage4)), [1, 64, 64, 1])

dhconv1 = tf.nn.relu(conv2d(dximage, W_conv1) + b_conv1)
dhpool1 = max_pool_2x2(dhconv1)

dhconv2 = tf.nn.relu(conv2d(dhpool1, W_conv2) + b_conv2)
dhpool2 = max_pool_2x2(dhconv2)

dhpool2flat = tf.reshape(dhpool2, [-1, 16 * 16 * 32])
dhfc1 = tf.nn.relu(tf.matmul(dhpool2flat, W_fc1) + b_fc1)

dhfc1drop = tf.nn.dropout(dhfc1, keep_prob)

dylog = tf.matmul(dhfc1drop, W_fc2) + b_fc2
dyconv = tf.nn.softmax(dylog)

rdhconv1 = tf.nn.relu(conv2d(rdximage, W_conv1) + b_conv1)
rdhpool1 = max_pool_2x2(rdhconv1)

rdhconv2 = tf.nn.relu(conv2d(rdhpool1, W_conv2) + b_conv2)
rdhpool2 = max_pool_2x2(rdhconv2)

rdhpool2flat = tf.reshape(rdhpool2, [-1, 16 * 16 * 32])
rdhfc1 = tf.nn.relu(tf.matmul(rdhpool2flat, W_fc1) + b_fc1)

rdhfc1drop = tf.nn.dropout(rdhfc1, keep_prob)

rdylog = tf.matmul(rdhfc1drop, W_fc2) + b_fc2
rdyconv = tf.nn.softmax(rdylog)

dl2loss =  tf.nn.l2_loss(dximage - rdximage)# + tf.nn.l2_loss(dhpool1 - rdhpool1)

dyloss = -tf.reduce_sum(y_*tf.log(dyconv))
dhfc1loss = tf.nn.l2_loss(tf.sub(dhfc1, hfc1))
rdyloss = -tf.reduce_sum(y_*tf.log(rdyconv))
#
dloss = dyloss + dhfc1loss + Floss + Dloss#rdyloss #+ dl2loss
sess.run(tf.initialize_variables(set(tf.all_variables()) - firstNetworkVars))


dtrain_step = tf.train.AdamOptimizer(1e-2).minimize(dloss, var_list = [dximage])

#%%

sess.run(tf.initialize_variables(set(tf.all_variables()) - firstNetworkVars))
batch_size = 50
first = dximage.eval()[0, :, :, 0]
ces = []
l2 = 0
l23 = 0
ce = 0
for i in range(20000):
    if i % 100 == 0 and i > 0:
        plt.plot(ces)
        plt.show()
        c = dximage.eval()[0, :, :, 0]
        plt.imshow(c, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.colorbar()
        plt.show()
        print 'fwd', ce, l2, l23, l24

    train, ce, l2, l23, l24 = sess.run([dtrain_step, dyloss, Floss, Dloss, dhfc1loss], feed_dict = { y_: [[1.0, 0.0]], keep_prob: 1.0 })

    print 'fwd', ce, l2, l23, l24

    ces.append(ce)
    #1/0

    #train = sess.run([train_deconv_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: batch, y_: batch_y, keep_prob: 1.0}))


#%%
import skimage.io

signal = skimage.io.imread('/home/bbales2/microhog/rafting_rotated_2d/a1h/0/signalx.png').astype('float')

signal = signal[0:64, 0:64]

signal -= signal.flatten().min()
signal /= signal.flatten().max()
signal -= 0.5#numpy.mean(signal)
signal *= 2.0

#%%
tf.assign(dximage, signal.reshape([1, 64, 64, 1]))
#%%
plt.imshow(signal)
plt.colorbar()
plt.show()
plt.imshow(rdximage.eval()[0, :, :, 0])
plt.colorbar()
plt.show()
#%%
t = fftc.eval()
#%%
W_log = weight_variable([2, 128])

h_fc1_rep = tf.matmul(y_conv, W_log)

W_log2 = weight_variable([128, 15 * 20 * 32])

h_pool2_rep = tf.reshape(tf.matmul(h_fc1_rep, W_log2), [-1, 15, 20, 32])
h_conv2_rep = tf.mul(h_conv1_mask, tf.image.resize_nearest_neighbor(h_pool2_rep, [60, 80]))

#h_conv1_rep = tf.image.resize_nearest_neighbor(h_pool1_rep, [28, 28]))

W_deconv2 = weight_variable([5, 5, 32, 1])
b_deconv2 = bias_variable([1])

x_rep = tf.nn.conv2d(h_conv2_rep, W_deconv2, strides = [1, 1, 1, 1], padding='SAME') + b_deconv2

h_fc1_rep_loss = tf.nn.l2_loss(h_fc1_rep - h_fc1)
h_pool1_loss = tf.nn.l2_loss(h_pool2_rep - h_pool1)
x_loss = tf.nn.l2_loss(x_rep - x_image)
rep_loss = x_loss + h_pool1_loss
train_deconv_step = tf.train.AdamOptimizer(1e-2).minimize(x_loss + h_pool1_loss,#h_fc1_rep_loss +
              var_list = [W_log, W_log2, W_deconv2, b_deconv2])
#%%
y_conv_gen = tf.placeholder(tf.float32, shape=[None, 2])
h_conv1_mask_gen = tf.placeholder(tf.float32, shape=[None, 60, 80, 32])

h_fc1_gen = tf.placeholder(tf.float32, shape=[None, 128])
#h_fc1_gen = tf.matmul(y_conv_gen, W_log)

h_pool2_gen = tf.reshape(tf.matmul(h_fc1_gen, W_log2), [-1, 15, 20, 32])
#h_pool2_gen = tf.placeholder(tf.float32, shape=[None, 15, 20, 32])
h_conv2_gen = tf.mul(h_conv1_mask_gen, tf.image.resize_nearest_neighbor(h_pool2_gen, [60, 80]))

x_gen = tf.nn.conv2d(h_conv2_gen, W_deconv2, strides = [1, 1, 1, 1], padding='SAME') + b_deconv2

#%%
xs = x_gen.eval(feed_dict = {y_conv_gen : [[7.5, 5.6]], h_conv1_mask_gen : hc1m[1:2, :, :, :], h_fc1_gen : numpy.random.randn(1, 128)})

plt.imshow(xs[0, :, :, 0], cmap = plt.cm.gray)
plt.show()
#%%
acc, ce, yc, yl, hc1m, xr2 = sess.run([accuracy, cross_entropy, y_conv, y_log, h_conv1_mask, x_rep], feed_dict={ x: [im.flatten() for im in Xtest], y_: Ytest * 1, keep_prob: 1.0 })
print acc, ce
#%%
test = hc1m[0, :, :, 0]
output = numpy.zeros(test.shape)
compressed = numpy.zeros((test.shape[0] / 4, test.shape[1] / 4))

for i in range(0, test.shape[0] / 4):
    for j in range(0, test.shape[1] / 4):
        idx = numpy.argmax(test[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4].flatten())
        ix = idx % 4
        iy = idx / 4

        compressed[i, j] = idx
        output[i * 4 + iy, j * 4 + ix] = 1

plt.imshow(compressed, interpolation = 'NONE', cmap = plt.cm.gray)
plt.show()
plt.imshow(test, interpolation = 'NONE')
plt.show()
plt.imshow(output, interpolation = 'NONE')
plt.show()
#%%
for im, yref, y in zip(Xtest, Ytest * 1, yc):
    if numpy.argmax(yref) != numpy.argmax(y):
        plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
        plt.show()

        print yref, y
        print '----'