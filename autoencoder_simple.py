#%%

import os
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/autoencoder')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#h_exp1 = tf.exp(tf.mul(h_conv1, beta))

#h_pool1_avg = tf.nn.avg_pool(tf.mul(h_conv1, h_exp1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#h_conv1_avg = tf.nn.avg_pool(h_exp1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#h_pool1 = tf.div(h_pool1_avg, h_conv1_avg)

#%%

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

#def weight_variable(shape):
#  initial = tf.truncated_normal(shape, stddev=0.1)
#  return tf.Variable(initial)

#def bias_variable(shape):
#  initial = tf.constant(0.1, shape=shape)
#  return tf.Variable(initial)

#def conv2d(x, W):
#  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#W_conv1 = weight_variable([5, 5, 1, 32])
#b_conv1 = bias_variable([32])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.zeros([10, 784]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
x2 = tf.nn.softmax(tf.matmul(y, W2))

y_ = tf.placeholder(tf.float32, [None, 10])
yloss = tf.reduce_sum(y_*tf.log(y))
xloss = -tf.nn.l2_loss(x - x2)
cross_entropy = -yloss - xloss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if i % 100 == 0:
      acc, yloss2, xloss2, W22, x22 = sess.run([accuracy, yloss, xloss, W2, x2], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      plt.imshow(x22[0].reshape(28, 28))
      plt.show()
      plt.imshow(mnist.test.images[0].reshape(28, 28))
      plt.show()
      print acc, yloss2, xloss2

#%%