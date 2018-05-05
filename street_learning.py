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

EPOCHS = 50
BATCH_SIZE = 8

class StreetLearning:
    def __init__(self):
        #dataset
        dataset_name = 'kitti_data_road' #'einstein'
        self.train_img_path = 'data/' + dataset_name + '/train/inputs'
        self.train_target_img_path = 'data/' + dataset_name + '/train/targets'
        self.test_img_path = 'data/' + dataset_name + '/test/inputs'
        self.test_target_img_path = 'data/' + dataset_name + '/test/targets'

        self.input_dim =  [320, 1024,3]#[375,1242,3]#[28,28,3] #this must be the right size of the images
        self.resized_dim = [80, 256]#[28,92]#[28,92]#[28,28]
        #model
        self.keep_prob = tf.constant(0.75)
        self.training = False
        #label
        self.color_table = [[255,0,0],      # no_street
                            [255,0,255],    # street
                            [0,0,0],
                            [200,200,200]]        # something black
        self.n_classes = len(self.color_table)

    def get_dataset(self, img_path, target_img_path):
        '''
        Parameters: paths to the dataset folder
        Return: tuple of 2 tensors:
            filenames: list of paths of each image in the dataset
            labels: list of labels
        '''
        inputs_file_paths = glob.glob(os.path.join(img_path, '*'))
        labels = []
        for input_file_path in inputs_file_paths:
            #Extract target filepath
            '''file_header = input_file_path.split("/")[-1].split("_")[0]
            file_number = input_file_path.split("_")[-1]
            target_file_path = target_img_path + "/" + file_header + '_'
            if file_header == 'um':
                target_file_path += 'lane_'
            else:
                target_file_path += 'road_' '''
            file_number = input_file_path.split("/")[-1]
            #file_number = input_file_path.split("_")[-1]
            target_file_path = target_img_path + '/' + file_number
            labels.append(target_file_path)  #np.random.sample(self.n_classes) #[random.randint(0,self.n_classes-1)]*self.n_classes

        filenames = tf.constant(inputs_file_paths)
        labels = tf.constant(labels)
        return (filenames, labels)

    def encode_label(self, semantic_map):
        palette = tf.constant(self.color_table, dtype=tf.uint8)
        class_indexes = tf.argmax(semantic_map, axis=-1)
        class_indexes = tf.reshape(class_indexes, [-1])
        color_image = tf.gather(palette, class_indexes)
        color_image = tf.reshape(color_image, [self.resized_dim[0], self.resized_dim[1], 3])
        return color_image

    def decode_label(self, label):
        semantic_map = []
        for colour in self.color_table:
            class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
            semantic_map.append(class_map)
        semantic_map = tf.stack(semantic_map, axis=-1)
        semantic_map = tf.cast(semantic_map, tf.float32)
        return semantic_map

    def _parse_function(self, filename, label_path):
        # image
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string)  #image_decoded = tf.image.decode_png(image_string)
        image_reshaped = tf.reshape(image_decoded, self.input_dim)
        image_resized = tf.image.resize_images(image_reshaped, self.resized_dim)
        image = tf.image.convert_image_dtype(image_resized, dtype = tf.float32)
        # label
        label_string = tf.read_file(label_path)
        label_decoded = tf.image.decode_jpeg(label_string)  #image_decoded = tf.image.decode_png(image_string)
        label_reshaped = tf.reshape(label_decoded, self.input_dim)
        label_resized = tf.image.resize_images(label_reshaped, self.resized_dim)
        label = self.decode_label(label_resized)
        return image, label

    def get_data(self):
        # using two numpy arrays
        train_data = self.get_dataset(self.train_img_path, self.train_target_img_path)   # tuple of (inputs filenames, labels)
        test_data = self.get_dataset(self.test_img_path, self.test_target_img_path)  # tuple of (inputs filenames, labels)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
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
        # TODO SOFTMAX
        #self.segmentation_result = tf.nn.softmax(deconv2)
        #self.logits = tf.layers.dense(deconv3, self.n_classes, name='logits')
        #self.segmentation_result = tf.sigmoid(deconv3)

    def loss(self):
        '''
        Loss function
        '''
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.labels)))
        #entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.segmentation_result)
        #self.loss = tf.reduce_mean(entropy, name='loss')

    def optimizer(self):
        '''
        Optimizer
        '''
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self):
        image_int32 = tf.image.convert_image_dtype(self.segmentation_result[0], dtype = tf.int32)
        img_encoded = self.encode_label(image_int32)
        image_uint8 = tf.image.convert_image_dtype(img_encoded, dtype = tf.uint8)
        img = tf.image.encode_jpeg(image_uint8)
        save_img_op = tf.write_file('data/seg_out.jpeg',img)
        with tf.Session() as sess:
            print('in the session')
            #print('The shape:', sess.run(tf.shape(self.thing_we_want_the_shape_of)))
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
                sess.run(save_img_op)

            # initialise iterator with test data
            #sess.run(self.test_init)
            #self.training = False
            #print('Test Loss: {:4f}'.format(sess.run(self.loss)))
            #sess.run(save_img_op)

    def testtest(self):
        img = scipy.misc.imread('data/kitti_data_road/train/targets/0.png', mode = 'RGB')
        label_reshaped = tf.reshape(img, self.input_dim)
        label_resized = tf.image.resize_images(label_reshaped, self.resized_dim)
        #scipy.misc.imsave('data/testtest.png', img)
        print('***fatto')
        #decode
        semantic_map = self.decode_label(label_resized)
        #encode
        color_image = self.encode_label(semantic_map)


        with tf.Session() as sess:
            #magic_number_val = sess.run(magic_number)
            color_image_val = sess.run(color_image)
            scipy.misc.imsave('data/testtest.png', color_image_val)
            '''train_len = len(glob.glob(os.path.join(self.train_img_path, '*')))
            n_batches = train_len // BATCH_SIZE
            sess.run(self.train_init)
            for _ in range(n_batches):
                sess.run(save_img_op)
                input('go to next')'''

        print('LABEL')



if __name__ == '__main__':
    sl = StreetLearning()
    sl.get_data()
    sl.model()
    sl.loss()
    sl.optimizer()
    sl.train()
    #sl.testtest()
