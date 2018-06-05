import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np

keep_prob = tf.constant(0.75)
training = True
n_classes = 5
input_dim = [8,8,3]

EPOCHS = 10
BATCH_SIZE = 16
# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)

shape = [None]+input_dim
x, y = tf.placeholder(tf.float32, shape= shape ), tf.placeholder(tf.float32, shape=[None,n_classes])
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()

# using two numpy arrays
train_shape = [100] + input_dim
train_shape = tuple(train_shape)
test_shape = [20] + input_dim
test_shape = tuple(test_shape)
train_data = (np.random.sample(train_shape), np.random.sample((100,n_classes)))
test_data = (np.random.sample(test_shape), np.random.sample((20,n_classes)))

n_batches = len(train_data[0]) // BATCH_SIZE

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
# make a simple model
'''
net = tf.layers.dense(features, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, n_classes, activation=tf.tanh)
'''
conv1 = tf.layers.conv2d(inputs=features,
                          filters=32,
                          kernel_size=[5, 5],
                          padding='SAME',
                          activation=tf.nn.relu,
                          name='conv1')
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2],
                                strides=2,
                                name='pool1')
conv2 = tf.layers.conv2d(inputs=pool1,
                          filters=64,
                          kernel_size=[5, 5],
                          padding='SAME',
                          activation=tf.nn.relu,
                          name='conv2')
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2],
                                strides=2,
                                name='pool2')
feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
pool2 = tf.reshape(pool2, [-1, feature_dim])
fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')
dropout = tf.layers.dropout(fc,
                            keep_prob,
                            training=training,
                            name='dropout')
prediction = tf.layers.dense(dropout, n_classes, name='logits') #self.logits = tf.layers.dense(dropout, n_classes, name='logits')

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=prediction)
loss = tf.reduce_mean(entropy, name='loss')
#loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    print('Training...')
    for i in range(EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
    print('Test Loss: {:4f}'.format(sess.run(loss)))
