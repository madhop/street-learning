import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import kernels
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

import tensorflow as tf

img_path = 'data/kitti_data_road/trainingtraining/inputs'

def read_one_image(filename):
    ''' This method is to show how to read image from a file into a tensor.
    The output is a tensor object.
    '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image = tf.cast(image_decoded, tf.float32) / 256.0
    print(filename)
    return image

def show_images(images, rgb=True):
    gs = gridspec.GridSpec(1, len(images))
    for i, image in enumerate(images):
        plt.subplot(gs[0, i])
        if rgb:
            plt.imshow(image)
        else:
            image = image.reshape(image.shape[0], image.shape[1])
            plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()

file_paths = glob.glob(os.path.join(img_path, '*'))

kernels_list = [kernels.BLUR_FILTER_RGB,
                kernels.SHARPEN_FILTER_RGB,
                kernels.EDGE_FILTER_RGB,
                kernels.TOP_SOBEL_RGB,
                kernels.EMBOSS_FILTER_RGB]
kernels_list = kernels_list[1:]

for file_path in file_paths:
    image = read_one_image(file_path)
    image = tf.expand_dims(image, 0)
    images = [image[0]]
    with tf.Session() as sess:
        images = sess.run(images)
    show_images(images)
