import glob
from PIL import Image
import os
from skimage.io import imsave, imread
from skimage.measure import block_reduce
import numpy as np

def crop_center(img,cropx,cropy):
   y,x,z = img.shape
   startx = x//2-(cropx//2)
   starty = y//2-(cropy//2)
   return img[starty:starty+cropy,startx:startx+cropx]

dataset_name = 'kitti_data_road' #'einstein'
img_path = 'data/' + dataset_name + '/training/inputs'

image_cols_origin = 1024
image_rows_origin = 320


inputs_file_paths = glob.glob(os.path.join(img_path, '*'))

i = 0
for image_name in inputs_file_paths:
   print (i)
   #image_name
   img = imread(os.path.join(image_name), as_grey=False)
   img = crop_center(img,image_cols_origin,image_rows_origin)
   #img = block_reduce(img, block_size=(4, 4, 1), func=np.mean)
   img = Image.fromarray(img, 'RGB')
   img.save('data/kitti_data_road/train/inputs/'+str(i)+'.png')
   i += 1

print('salvate!')
