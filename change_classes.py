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
            out = np.zeros(np.shape(img))
            out[:,:,3] = 255
            out[(img[:,:,0] == 128) & (img[:,:,1] == 64) & (img[:,:,2] == 128)] = [255,255,255,255]
            out = out.astype(np.uint8)
            os.remove(image_name)
            img = Image.fromarray(out, 'RGBA')
            img.save(image_name)

print('salvate!')
