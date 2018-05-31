#from tensorflow.examples.tutorials.mnist import input_data
#MNIST = input_data.read_data_sets('data/mnist', one_hot=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
n_classes = 2
learning_rate = 0.01
batch_size = 3#128
n_epochs = 30
n_train = 60000
n_test = 10000

# Read data
inputs_folder = 'data/kitti_data_road/trainingtraining/inputs'
test_folder = 'data/kitti_data_road/testingtesting/tests'
#train, val, test = utils.read_kitti_data_road(kitti_data_road_folder)
train = utils.read_kitti_data_road(inputs_folder)

# Create dataset
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # optional
train_data = train_data.batch(batch_size)

'''test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)'''

# Create iterators
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
'''test_init = iterator.make_initializer(test_data)'''

# Build model to predict
#logits = tf.matmul(img, w) + b
fc = tf.layers.dense(img, 8, activation=tf.nn.relu, name='fc')
fc1 = tf.layers.dense(fc, 8, activation=tf.nn.relu, name='fc1')
logits = tf.layers.dense(fc1, n_classes, name='logits')

# loss function
#loss = tf.losses.mean_squared_error(logits, label)
entropy = tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logits, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    #test model
    '''sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))'''
writer.close()
print('TensorFlow')
