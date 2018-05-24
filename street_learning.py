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

EPOCHS = 40
BATCH_SIZE = 16

class StreetLearning:
    def __init__(self):
        #optimizer
        self.learning_rate=0.01
        #dataset
        dataset_name = 'kitti_data_road' #'einstein'
        self.train_img_path = 'data/' + dataset_name + '/train/inputs'
        self.train_target_img_path = 'data/' + dataset_name + '/train/targets'
        self.test_img_path = 'data/' + dataset_name + '/test/inputs'
        self.test_target_img_path = 'data/' + dataset_name + '/test/targets'

        self.input_dim =  [320, 1024,3]#[375,1242,3]#[28,28,3] #this must be the right size of the images
        self.resized_dim = [320, 1024]#[80, 256]#[160, 512]#[28,92]#[28,92]#[28,28]
        #model
        self.n_layers = 5
        self.keep_prob = tf.constant(0.75)
        self.training = False
        self.num_filters = 3
        self.kernel = [5, 5]
        #label
        self.color_table = [[255,0,0],      # no_street
                            [255,0,255],    # street
                            [0,0,0]]        # something black
        self.n_classes = len(self.color_table)

        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            #tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

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
        label_decoded = tf.image.decode_png(label_string)  #image_decoded = tf.image.decode_png(image_string)
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
        encode_model = {}
        for l in range(self.n_layers):
            if l == 0:
                layer_input = self.features
            else:
                layer_input = encode_model['pool'+str(l-1)]
            conv1 = tf.layers.conv2d(inputs=layer_input,
                                      filters= 2**(self.num_filters + l),
                                      kernel_size=self.kernel,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      trainable=self.training,
                                      name='conv'+str(l)+'_1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                                      filters=2**(self.num_filters + l),
                                      kernel_size=self.kernel,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      trainable=self.training,
                                      name='conv'+str(l)+'_2')
            encode_model['conv'+str(l)] = conv2
            pool = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool'+str(l))
            encode_model['pool'+str(l)] = pool


        conv1 = tf.layers.conv2d(inputs=pool,
                                  filters=2**(self.num_filters + (l+1)),
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  trainable=self.training,
                                  name='conv'+str(l+1)+'_1')
        conv2 = tf.layers.conv2d(inputs=conv1,
                                  filters=2**(self.num_filters + (l+1)),
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  trainable=self.training,
                                  name='conv'+str(l+1)+'_2')
        # Decode
        decode_model = {}
        for l in range(self.n_layers-1,-1,-1):#range(self.n_layers):
            print(l)
            if l == self.n_layers-1:
                layer_input = conv2
            else:
                layer_input = decode_model[str(l+1)]
            print('input',layer_input)
            unpool_temp = tf.layers.conv2d_transpose(inputs=layer_input,
                                                filters=2**(self.num_filters + l),
                                                kernel_size=[2, 2],
                                                strides=(2,2),
                                                padding='SAME',
                                                activation=tf.nn.relu,
                                                name='unpool'+str(l))
            unpool = tf.concat([encode_model['conv'+str(l)], unpool_temp], 3, name='concat'+str(l))
            deconv1 = tf.layers.conv2d(inputs=unpool,
                                      filters=2**(self.num_filters + l),
                                      kernel_size=self.kernel,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      trainable=self.training,
                                      name='deconv'+str(l)+'_1')
            deconv2 = tf.layers.conv2d(inputs=deconv1,
                                      filters=2**(self.num_filters + l),
                                      kernel_size=self.kernel,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      trainable=self.training,
                                      name='deconv'+str(l)+'_2')
            decode_model[str(l)] = deconv2
            print('output',deconv2)

        # Sigmoid
        self.segmentation_result = tf.layers.conv2d(inputs=deconv2,
                                  filters=self.n_classes,
                                  kernel_size=[1, 1],
                                  padding='SAME',
                                  activation=tf.sigmoid,
                                  trainable=self.training,
                                  name='sigmoid')
        #Softmax
        #self.segmentation_result = tf.nn.softmax(sigmoid)

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
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.gstep)

    def train(self):
        image_int32 = tf.image.convert_image_dtype(self.segmentation_result[0], dtype = tf.int32)
        img_encoded = self.encode_label(image_int32)
        image_uint8 = tf.image.convert_image_dtype(img_encoded, dtype = tf.uint8)
        img = tf.image.encode_jpeg(image_uint8)

        writer = tf.summary.FileWriter('./graphs/street_learning', tf.get_default_graph())

        with tf.Session() as sess:
            print('in the session')
            sess.run(tf.global_variables_initializer())

            self.training = True

            train_len = len(glob.glob(os.path.join(self.train_img_path, '*')))
            n_batches = train_len // BATCH_SIZE

            # initialise iterator with train data
            sess.run(self.train_init)

            print('Training...')
            start = time.time()
            for i in range(EPOCHS):
                tot_loss = 0
                step = self.gstep.eval()
                for _ in range(n_batches):
                    _, loss_value, summaries = sess.run([self.train_op, self.loss, self.summary_op])
                    writer.add_summary(summaries, global_step=step)
                    tot_loss += loss_value
                print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
                print('Until now, it took', time.time()-start, 's')
                #save epoch result
                save_epoch_result_path = 'data/outputs/' + str(i) + '_seg_out.jpeg'
                save_training_img_op = tf.write_file(save_epoch_result_path,img)
                sess.run(save_training_img_op)
            print('Training took', time.time()-start, 's')

            # initialise iterator with test data
            sess.run(self.test_init)
            self.training = False

            print('Test Loss: {:4f}'.format(sess.run(self.loss)))
            # save test result
            save_epoch_result_path = 'data/outputs/test_seg_out.jpeg'
            save_test_img_op = tf.write_file(save_epoch_result_path,img)
            sess.run(save_test_img_op)

        writer.close()

    def testtest(self):
        img = scipy.misc.imread('data/kitti_data_road/train/targets/0.png', mode = 'RGB')
        label_reshaped = tf.reshape(img, self.input_dim)
        label_resized = tf.image.resize_images(label_reshaped, self.resized_dim)
        print('***testtest')
        #decode
        semantic_map = self.decode_label(label_resized)
        #encode
        color_image = self.encode_label(semantic_map)
        with tf.Session() as sess:
            color_image_val = sess.run(color_image)
            scipy.misc.imsave('data/testtest.png', color_image_val)

        print('LABEL')



if __name__ == '__main__':
    sl = StreetLearning()
    sl.get_data()
    sl.model()
    sl.loss()
    sl.optimizer()
    sl.summary()
    sl.train()
    #sl.testtest()
