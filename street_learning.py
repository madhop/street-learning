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

class StreetLearning:
    def __init__(self):
        #dataaset
        dataset_name = 'einstein' #'kitti_data_road'
        self.train_img_path = 'data/' + dataset_name + '/trainingtraining/inputs'
        self.train_target_img_path = 'data/' + dataset_name + '/trainingtraining/targets'
        self.test_img_path = 'data/' + dataset_name + '/testingtesting/inputs'
        self.test_target_img_path = 'data/' + dataset_name + '/testingtesting/targets'

        self.input_dim = [28,28,3] #[1242,375,3] #this must be the right size of the images
        self.resized_dim = [28,28]#[92,28]
        #model
        self.keep_prob = tf.constant(0.75)
        self.training = False
        self.n_classes = 2

    def get_dataset(self, img_path, target_img_path):
        '''
        Parameters: paths to the dataset folder
        Return: tuple of 2 tensors:
            filenames: list of paths of each image in th edataset
            labels: list of labels
        '''
        inputs_file_paths = glob.glob(os.path.join(img_path, '*'))
        labels = []
        for input_file_path in inputs_file_paths:
            #Extract target filepath
            file_header = input_file_path.split("/")[-1].split("_")[0]
            file_number = input_file_path.split("_")[-1]
            target_file_path = target_img_path + "/" + file_header + '_'
            if file_header == 'um':
                target_file_path += 'lane_'
                labels.append([1,0])
            else:
                labels.append([0,1])
                target_file_path += 'road_'
            target_file_path += file_number
            #labels.append([1]*self.n_classes)#(target_file_path)  #np.random.sample(self.n_classes) #[random.randint(0,self.n_classes-1)]*self.n_classes

        filenames = tf.constant(inputs_file_paths)
        labels = tf.constant(labels)
        return (filenames, labels)

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)  #image_decoded = tf.image.decode_png(image_string)
        image_reshaped = tf.reshape(image_decoded, self.input_dim)
        image_resized = tf.image.resize_images(image_reshaped, self.resized_dim)
        image_float32 = tf.image.convert_image_dtype(image_resized, dtype = tf.float32)
        return image_float32, label

    def get_data(self):
        # using two numpy arrays
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
        '''
        Function to build the neural net
        '''
        # Encode
        conv1 = tf.layers.conv2d(inputs=self.features,
                                  filters=32,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation=tf.nn.relu,
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
                                  name='conv3')
        # Decode
        #unpool1 = tf.keras.layers.UpSampling2D(size = (conv3.shape[1]*2, conv3.shape[2]*2))(conv3)
        unpool1 = tf.layers.conv2d_transpose(inputs=conv3,
                                            filters=128,    #TODO conv2d_transpose o conv2d
                                            kernel_size=[2, 2],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            activation=tf.nn.relu,
                                            name='unpool1')
        deconv1 = tf.layers.conv2d(inputs=unpool1,
                                  filters=64,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='deconv1')
        unpool2 = tf.layers.conv2d_transpose(inputs=deconv1,
                                            filters=64,
                                            kernel_size=[2, 2],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            activation=tf.nn.relu,
                                            name='unpool2')
        deconv2 = tf.layers.conv2d(inputs=unpool2,
                                  filters=32,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='deconv2')
        deconv3 = tf.layers.conv2d(inputs=deconv2,
                                  filters=1,
                                  kernel_size=[1, 1],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='deconv3')
        # TODO SOFTMAX
        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    def loss(self):
        '''
        Loss function
        '''
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_mean(entropy, name='loss')

    def optimizer(self):
        '''
        Optimizer
        '''
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        with tf.Session() as sess:
            print('in the session')
            self.training = True
            train_len = len(glob.glob(os.path.join(self.train_img_path, '*')))
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
            self.training = False
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
