import glob
from PIL import Image
import os
from skimage.io import imsave, imread
from skimage.measure import block_reduce
from skimage.viewer import ImageViewer
import numpy as np

train_test = 'val'

sets = 'data/CityScapes/gtFine/' + train_test #gtFine #leftImg8bit
cities_folders = glob.glob(os.path.join(sets, '*'))
save_in = 'data/CityScapes/' + 'test' + '/targets/'

i = 0
for city_folder in cities_folders:
    im_file_paths = glob.glob(os.path.join(city_folder, '*'))
    i+=1
    print(i, city_folder)
    for image_name in im_file_paths:
        img = imread(os.path.join(image_name), as_grey=False)
        img_save = Image.fromarray(img, 'RGB')
        new_path = save_in + image_name.split('/')[-1]
        img_save.save(new_path)
