import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

constant_dim = 1
a = tf.constant([2]*constant_dim, name='a')
b = tf.constant(3, name='b')
x = tf.add(a,b, name='add')

writer = tf.summary.FileWriter('./graph', tf.get_default_graph())
l = []
for i in range(0,constant_dim):
    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        var = sess.run(x)
        print('x:',sess.run(x))
l.append(var)
writer.close()


print('l', l)

c = tf.constant(100, name='c')
add_op = tf.add(l, c)

with tf.Session() as sess:
    print('add_op:',sess.run(add_op))
