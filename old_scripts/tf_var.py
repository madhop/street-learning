import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

s = tf.get_variable("scalar", initializer = tf.constant(2))
m = tf.get_variable("matrix", initializer = tf.constant([[0,1],[2,3]]))
W = tf.get_variable("big_matriz", shape = (784, 10), initializer = tf.zeros_initializer())

with tf.Session() as sess:
    #sess.run(tf.variables_initializer([s, m]))
    sess.run(s.initializer)
    print(sess.run(s))
