import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np

EPOCHS = 10
BATCH_SIZE = 16
# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)

class StreetLearning:
    def __init__(self):
        self.keep_prob = tf.constant(0.75)
        self.training = True
        self.n_classes = 5
        self.input_dim = [8,8,3]

    def get_data(self):
        '''shape = [None]+self.input_dim
        self.x, self.y = tf.placeholder(tf.float32, shape = shape ), tf.placeholder(tf.float32, shape=[None,self.n_classes])
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(batch_size).repeat()'''

        # using two numpy arrays
        train_shape = [100] + self.input_dim
        train_shape = tuple(train_shape)
        test_shape = [20] + self.input_dim
        test_shape = tuple(test_shape)
        train = (np.random.sample(train_shape).astype(np.float32), np.random.sample((100,self.n_classes)).astype(np.float32))
        self.train_len = len(train[0])
        test = (np.random.sample(test_shape).astype(np.float32), np.random.sample((20,self.n_classes)).astype(np.float32))

        train_data = tf.data.Dataset.from_tensor_slices(train).shuffle(10000).batch(BATCH_SIZE).repeat()
        test_data = tf.data.Dataset.from_tensor_slices(test).batch(test[0].shape[0]).repeat()

        #iterator
        iterator = tf.data.Iterator.from_structure(train_data.output_types,train_data.output_shapes)
        self.features, self.labels = iterator.get_next()

        #init
        self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
        self.test_init = iterator.make_initializer(test_data)    # initializer for test_data

    def model(self):
        conv1 = tf.layers.conv2d(inputs=self.features,
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
                                    self.keep_prob,
                                    training=self.training,
                                    name='dropout')
        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    def loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_mean(entropy, name='loss')

    def optimizer(self):
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        n_batches = self.train_len // BATCH_SIZE
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialise iterator with train data
            sess.run(self.train_init)
            print('Training...')
            for i in range(EPOCHS):
                tot_loss = 0
                for _ in range(n_batches):
                    _, loss_value = sess.run([self.train_op, self.loss])
                    tot_loss += loss_value
                print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
            # initialise iterator with test data
            sess.run(self.test_init)
            print('Test Loss: {:4f}'.format(sess.run(self.loss)))


if __name__ == '__main__':
    sl = StreetLearning()
    sl.get_data()
    sl.model()
    sl.loss()
    sl.optimizer()
    sl.train()
