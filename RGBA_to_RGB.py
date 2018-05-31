import glob
from PIL import Image
import os
from skimage.io import imsave, imread
from skimage.measure import block_reduce
from skimage.viewer import ImageViewer
import numpy as np

sets = 'data/CityScapes/gtFine/'
sets_paths = glob.glob(os.path.join(sets, '*'))

i = 0
for set in sets_paths:
    cities_folders = glob.glob(os.path.join(set, '*'))
    for city_folder in cities_folders:
        im_file_paths = glob.glob(os.path.join(city_folder, '*'))
        im_file_paths.sort()
        i+=1
        print(i, city_folder)
        for image_name in im_file_paths:
            img = imread(os.path.join(image_name), as_grey=False)
            img = img[:,:,0:3]
            os.remove(image_name)
            img_save = Image.fromarray(img, 'RGB')
            img_save.save(image_name)
