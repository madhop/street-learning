import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time

import tensorflow as tf

img_path = './data/kitti_data_road/trainingtraining/inputs'

file_paths = glob.glob(os.path.join(img_path, '*'))
for file_path in file_paths:
    print(file_path)
