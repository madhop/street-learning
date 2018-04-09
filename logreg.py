#from tensorflow.examples.tutorials.mnist import input_data
#MNIST = input_data.read_data_sets('data/mnist', one_hot=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Read data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Create dataset
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # optional
train_data = train_data.batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

# Create iterators
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Create weights and biases
w = tf.get_variable('weights', shape = (784, 10),  initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable('biases', shape = (1,10), initializer=tf.zeros_initializer())

# Build model to predict
Y_predicted = 

print('TensorFlow')
