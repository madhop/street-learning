import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import random

EPOCHS = 10
BATCH_SIZE = 2#16
# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)

class StreetLearning:
    def __init__(self):
        #dataaset
        self.train_img_path = 'data/kitti_data_road/trainingtraining/inputs' #'data/einstein/trainingtraining/inputs'
        self.train_target_img_path = 'data/kitti_data_road/trainingtraining/targets' #'data/einstein/trainingtraining/targets'
        self.test_img_path = 'data/kitti_data_road/testingtesting/inputs' #'data/einstein/testingtesting/inputs'
        self.test_target_img_path = 'data/kitti_data_road/testingtesting/targets'#'data/einstein/testingtesting/targets'

        self.input_dim = [1242,375,3]  #[28,28,3] #this must be the right size of the images
        #model
        self.keep_prob = tf.constant(0.75)
        self.training = True
        self.n_classes = 5

    def get_dataset(self, img_path, target_img_path):
        inputs_file_paths = glob.glob(os.path.join(img_path, '*'))
        #data = []
        labels = []
        for input_file_path in inputs_file_paths:
            #Extract target filepath
            file_header = input_file_path.split("/")[-1].split("_")[0]
            file_number = input_file_path.split("_")[-1]
            target_file_path = target_img_path + "/" + file_header + '_'
            if file_header == 'um':
                target_file_path += 'lane_'
            else:
                target_file_path += 'road_'
            target_file_path += file_number
            #Load image
            #input_img = self.read_one_image(input_file_path)
            #target_img = self.read_one_image(target_file_path)
            #input_img = tf.expand_dims(input_img, 0)
            #target_img = tf.expand_dims(target_img, 0)
            #with tf.Session() as sess:
                #input_img, target_img = sess.run([input_img, target_img])
            #data.append(input_img)
            labels.append([1]*self.n_classes)#(target_file_path)  #np.random.sample(self.n_classes) #[random.randint(0,self.n_classes-1)]*self.n_classes

        #data = np.asarray(data)
        #labels = np.asarray(labels)

        filenames = tf.constant(inputs_file_paths)
        labels = tf.constant(labels)
        return (filenames, labels)

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        #image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.image.decode_jpeg(image_string)  #image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.reshape(image_decoded, self.input_dim)
        image_flaot32 = tf.image.convert_image_dtype(image_resized, dtype = tf.float32)
        return image_flaot32, label

    def get_data(self):
        # using two numpy arrays
        '''
        train_shape = [100] + self.input_dim
        train_shape = tuple(train_shape)
        test_shape = [20] + self.input_dim
        test_shape = tuple(test_shape)
        train = (np.random.sample(train_shape).astype(np.float32), np.random.sample((100,self.n_classes)).astype(np.float32))
        test = (np.random.sample(test_shape).astype(np.float32), np.random.sample((20,self.n_classes)).astype(np.float32))
        '''
        self.train_data = self.get_dataset(self.train_img_path, self.train_target_img_path)   # tuple of (inputs filenames, labels)
        test_data = self.get_dataset(self.test_img_path, self.test_target_img_path)  # tuple of (inputs filenames, labels)

        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
        train_dataset = train_dataset.map(self._parse_function).batch(BATCH_SIZE).repeat()
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.map(self._parse_function).batch(test_data[0].shape[0]).repeat()

        #iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        self.features, self.labels = iterator.get_next()

        #init
        self.train_init = iterator.make_initializer(train_dataset)  # initializer for train_dataset
        self.test_init = iterator.make_initializer(test_dataset)    # initializer for test_dataset

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
        with tf.Session() as sess:
            print('in the session')
            train_len = 16#sess.run(tf.shape(self.train_data)[0])
            n_batches = train_len // BATCH_SIZE
            sess.run(tf.global_variables_initializer())
            print('variables initialized')
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
