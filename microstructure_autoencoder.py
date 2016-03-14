#%%

import os
import pickle
import sklearn.preprocessing
import numpy
import sklearn.cross_validation

os.chdir('/home/bbales2/autoencoder')

f = open('microstructure.pickle')
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

x = tf.placeholder(tf.float32, shape=[None, 4800])
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
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,60,80,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

beta = tf.Variable(10.0)

h_exp1 = tf.exp(tf.mul(h_conv1, beta))

h_pool1_avg = tf.nn.avg_pool(tf.mul(h_conv1, h_exp1), ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
h_conv1_avg = tf.nn.avg_pool(h_exp1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

out2 = tf.tile(h_conv1_avg, [1, 1, 4, 4])
out3 = tf.reshape(out2, [-1, 60, 80, 32])

h_conv1_mask = tf.mul(0.25**2, tf.div(h_exp1, out3))

h_pool1 = tf.div(h_pool1_avg, h_conv1_avg)
#W_conv2 = weight_variable([5, 5, 32, 64])
#b_conv2 = bias_variable([64])

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([15 * 20 * 32, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool1, [-1, 15*20*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 2])
b_fc2 = bias_variable([2])

y_log = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(y_log)

yloss = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-5))

W_log = weight_variable([2, 128])

h_fc1_rep = tf.matmul(y_conv, W_log)

W_log2 = weight_variable([128, 15 * 20 * 32])

h_pool2_rep = tf.reshape(tf.matmul(h_fc1_rep, W_log2), [-1, 15, 20, 32])
h_conv2_rep = tf.mul(h_conv1_mask, tf.image.resize_nearest_neighbor(h_pool2_rep, [60, 80]))

#h_conv1_rep = tf.image.resize_nearest_neighbor(h_pool1_rep, [28, 28]))

W_deconv2 = weight_variable([5, 5, 32, 1])
b_deconv2 = bias_variable([1])

x_rep = tf.nn.conv2d(h_conv2_rep, W_deconv2, strides = [1, 1, 1, 1], padding='SAME') + b_deconv2

yloss = -tf.reduce_sum(y_*tf.log(y_conv))
h_fc1_rep_loss = tf.nn.l2_loss(h_fc1_rep - h_fc1)
h_pool1_loss = tf.nn.l2_loss(h_pool2_rep - h_pool1)
x_loss = tf.nn.l2_loss(x_rep - x_image)

cross_entropy = yloss
rep_loss = x_loss + h_pool1_loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_deconv_step = tf.train.AdamOptimizer(1e-2).minimize(x_loss + h_pool1_loss,#h_fc1_rep_loss +
              var_list = [W_log, W_log2, W_deconv2, b_deconv2])
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%

sess.run(tf.initialize_all_variables())

batch_size = len(Xtrain) - 1
ces = []
for i in range(20000):
    start = 0#(i * batch_size) % len(Xtrain)
    stop = len(Xtrain)#((i + 1) * batch_size) % len(Xtrain)

    if start > stop:
        continue

    batch = [im.flatten() for im in Xtrain[start : stop]]
    batch_y = Ytrain[start : stop] * 1

  #if i%10 == 0:
  #  train_accuracy, yl, hl, p2, hp2, hp1, xl, xr, xi = sess.run([accuracy, yloss, h_fc1_rep_loss, h_pool2_loss, h_pool2_loss, h_pool1_loss, x_loss, x_rep, x_image], feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
  #  print("step %d, training accuracy %g"%(i, train_accuracy))
  #  plt.imshow(xr[0, :, :, 0], interpolation = 'NONE')
  #  plt.show()
  #  plt.imshow(xi[0, :, :, 0], interpolation = 'NONE')
  #  plt.show()
  #print yl, hl, p2, hp2, hp1, xl

  #if train_accuracy < 0.85:
#
    if i % 10 == 0:
        plt.plot(ces)
        plt.show()

    train, acc, ce = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: batch, y_: batch_y, keep_prob: 0.5})
    print 'fwd', acc, ce

    if acc > 0.8:
        train, ce, x_rep2 = sess.run([train_deconv_step, rep_loss, x_rep], feed_dict={x: batch, y_: batch_y, keep_prob: 1.0})
        ces.append(ce)
        print 'back', acc, ce

    #train = sess.run([train_deconv_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: batch, y_: batch_y, keep_prob: 1.0}))
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