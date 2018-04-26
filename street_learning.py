import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np


class StreetLearning:
    def __init__(self):
        self.lr = 0.001
        self.keep_prob = tf.constant(0.75)
        self.n_classes = 2
        self.training = False
        self.inputs_img_path = 'data/kitti_data_road/trainingtraining/inputs'
        self.target_img_path = 'data/kitti_data_road/trainingtraining/targets'
        self.rgb = True
        self.batch_size = 128

    def read_one_image(self, filename):
        ''' This method is to show how to read image from a file into a tensor.
        The output is a tensor object.
        '''
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image = tf.cast(image_decoded, tf.float32) / 256.0
        return image

    def show_images(self, images):
        gs = gridspec.GridSpec(1, len(images))
        for i, image in enumerate(images):
            plt.subplot(gs[0, i])
            if self.rgb:
                plt.imshow(image)
            else:
                image = image.reshape(image.shape[0], image.shape[1])
                plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

    def get_dataset(self):
        inputs_file_paths = glob.glob(os.path.join(self.inputs_img_path, '*'))
        train_data = []
        target_data = []
        for input_file_path in inputs_file_paths:
            #Extract target filepath
            file_header = input_file_path.split("/")[-1].split("_")[0]
            file_number = input_file_path.split("_")[-1]
            target_file_path = self.target_img_path + "/" + file_header + '_'
            if file_header == 'um':
                target_file_path += 'lane_'
            else:
                target_file_path += 'road_'
            target_file_path += file_number
            #Load image
            input_img = self.read_one_image(input_file_path)
            target_img = self.read_one_image(target_file_path)
            input_img = tf.expand_dims(input_img, 0)
            target_img = tf.expand_dims(target_img, 0)
            with tf.Session() as sess:
                input_img, target_img = sess.run([input_img, target_img])
            train_data.append(input_img)
            target_data.append(0)#(target_img)
            print('train_data', len(train_data))
            #self.label = 0
            #show_images([input_img[0],target_img[0]])

        # from dataset to tf.data
        train_data = np.asarray(train_data)
        target_data = np.asarray(target_data)
        print('get_dataset done')
        return (train_data, target_data)

    def get_data(self):
        train = self.get_dataset()
        train_data = tf.data.Dataset.from_tensor_slices(train)
        train_data = train_data.batch(self.batch_size)
        with tf.name_scope('data'):
            iterator = tf.data.Iterator.from_structure(train_data.output_types)#,train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[1, 375, 1242, 3]) #(1, 375, 1242, 3)
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            #self.test_init = iterator.make_initializer(test_data)    # initializer for train_data
        print('get_data done')

    def inference(self):
        print('image shape', np.shape(self.img))
        conv1 = tf.layers.conv2d(inputs=self.img,
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
        print('inference done')


    def train(self):
        print('start training')
        self.training = True
        #loss
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        self.loss = tf.reduce_mean(entropy, name='loss')
        # optimizer
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.Session() as sess:
            print('sess')
            sess.run(tf.global_variables_initializer())
            print("qui non si blocca")
            sess.run(self.train_init)
            print("qui non si blocca 2")
            total_loss = 0
            try:
                while True:
                    _, l = sess.run([self.opt, self.loss])
                    print('Loss: {0}'.format(step, l))
                    #writer.add_summary(summaries, global_step=step)
            except tf.errors.OutOfRangeError:
                pass



if __name__ == '__main__':
    sl = StreetLearning()
    sl.get_data()
    sl.inference()
    sl.train()
