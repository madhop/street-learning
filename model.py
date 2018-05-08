import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import random
import scipy.misc

def model(self):
    '''
    Function to build the neural net
    '''

    '''
    THIS SIMPLE MODLE WORKS
    '''
    # Encode
    conv1 = tf.layers.conv2d(inputs=self.features,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name='pool2')
    conv3 = tf.layers.conv2d(inputs=pool2,
                              filters=128,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv3')
    # Decode
    unpool1 = tf.layers.conv2d_transpose(inputs=conv3,
                                        filters=128,
                                        kernel_size=[2, 2],
                                        strides=(2,2),
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='unpool1')
    deconv1 = tf.layers.conv2d(inputs=unpool1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv1')
    unpool2 = tf.layers.conv2d_transpose(inputs=deconv1,
                                        filters=64,
                                        kernel_size=[2, 2],
                                        strides=(2,2),
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        trainable=self.training,
                                        name='unpool2')
    deconv2 = tf.layers.conv2d(inputs=unpool2,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv2')
    self.segmentation_result = tf.layers.conv2d(inputs=deconv2,
                              filters=self.n_classes,
                              kernel_size=[1, 1],
                              padding='SAME',
                              activation=tf.sigmoid,
                              trainable=self.training,
                              name='deconv3')
    '''# Encode
    # 1
    conv1_1 = tf.layers.conv2d(inputs=self.features,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv1_1')
    conv1_2 = tf.layers.conv2d(inputs=conv1_1,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv1_2')
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name='pool1')
    # 2
    conv2_1 = tf.layers.conv2d(inputs=pool1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv2_1')
    conv2_2 = tf.layers.conv2d(inputs=conv2_1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv2_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name='pool2')
    # 3
    conv3_1 = tf.layers.conv2d(inputs=pool2,
                              filters=128,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv3_1')
    conv3_2 = tf.layers.conv2d(inputs=pool2,
                              filters=128,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='conv3_2')
    # Decode
    # 1
    unpool1 = tf.layers.conv2d_transpose(inputs=conv3_2,
                                        filters=128,
                                        kernel_size=[2, 2],
                                        strides=(2,2),
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        name='unpool1')
    deconv1_1 = tf.layers.conv2d(inputs=unpool1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv1_1')
    deconv1_2 = tf.layers.conv2d(inputs=deconv1_1,
                              filters=64,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv1_2')
    # 2
    unpool2 = tf.layers.conv2d_transpose(inputs=deconv1_2,
                                        filters=64,
                                        kernel_size=[2, 2],
                                        strides=(2,2),
                                        padding='SAME',
                                        activation=tf.nn.relu,
                                        trainable=self.training,
                                        name='unpool2')
    deconv2_1 = tf.layers.conv2d(inputs=unpool2,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv2_1')
    deconv2_2 = tf.layers.conv2d(inputs=deconv2_1,
                              filters=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              activation=tf.nn.relu,
                              trainable=self.training,
                              name='deconv2_2')

    # Sigmoid and Softmax
    deconv3 = tf.layers.conv2d(inputs=deconv2_2,
                              filters=self.n_classes,
                              kernel_size=[1, 1],
                              padding='SAME',
                              activation=tf.sigmoid,
                              trainable=self.training,
                              name='deconv3')
    self.segmentation_result = tf.nn.softmax(deconv3)'''
