import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
import numpy as np

x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)

# create the iterator
#iter = dataset.make_one_shot_iterator()
iterator = tf.data.Iterator.from_structure(dataset.output_types)
el = iterator.get_next()
init = iterator.make_initializer(dataset)

b = tf.constant(3, dtype = 'float64', name='b')
#a = tf.constant(3, name='b')
op = tf.add(el,b, name='add')

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(op))



'''
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a,b, name='add')

writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print('x:',sess.run(x))
writer.close()
'''
