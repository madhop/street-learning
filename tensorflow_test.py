import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a,b, name='add')

writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print('x:',sess.run(x))
writer.close()
