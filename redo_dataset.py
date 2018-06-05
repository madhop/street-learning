import glob
from PIL import Image
import os
#from skimage.io import imsave, imread
#from skimage.measure import block_reduce
import numpy as np

image_cols_origin = 1024
image_rows_origin = 320
def crop_center(img,cropx,cropy):
   y,x,z = img.shape
   startx = x//2-(cropx//2)
   starty = y//2-(cropy//2)
   return img[starty:starty+cropy,startx:startx+cropx]


sets = 'data/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/'
sets_paths = glob.glob(os.path.join(sets, '*'))
i = 0
for set in sets_paths:
    cities_folders = glob.glob(os.path.join(set, '*'))
    for city_folder in cities_folders:
        im_file_paths = glob.glob(os.path.join(city_folder, '*'))
        im_file_paths.sort()
        for image_name in im_file_paths:
           print(i)
           os.rename(image_name, city_folder+'/'+str(i)+'.png')
           i += 1


sets = 'data/CityScapes/gtFine_trainvaltest/gtFine/'
sets_paths = glob.glob(os.path.join(sets, '*'))
i = 0
for set in sets_paths:
    cities_folders = glob.glob(os.path.join(set, '*'))
    for city_folder in cities_folders:
        im_file_paths = glob.glob(os.path.join(city_folder, '*'))
        im_file_paths.sort()
        for image_name in im_file_paths:
           print(i)
           if image_name.find('color') > 0 :
               os.rename(image_name, city_folder+'/'+str(i)+'.png')
               i += 1
           else:
               os.remove(image_name)

print('salvate!')
