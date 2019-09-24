import h5py as h5
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.io as sio
import scipy
from os.path import join


def save_images(path, image):
    return scipy.misc.imsave(path, image)


def get_image(image_path):

  path_to_image = image_path
  image_dirs = os.listdir(path_to_image)
  image_dirs.sort()

  train_images = np.full((len(image_dirs), 128, 128, 3), 0, dtype='float32')

  for i in range(len(image_dirs)):

    filename = image_dirs[i]

    # read image into arrays
    image_name = os.path.join(path_to_image, filename)

    img = mpimg.imread(image_name)
    img = 2*(np.float32(img)/255.0) -1
    train_images[i, :, :, :] = img

  print("shape of train_images {}".format(train_images.shape))
  return train_images


def get_mask(mask_path):

  path_to_mask = mask_path
  mask_dirs = os.listdir(path_to_mask)
  mask_dirs.sort()

  train_masks = np.full((len(mask_dirs), 128, 128, 1), 0, dtype='float32')

  for i in range(len(mask_dirs)):

    filename = mask_dirs[i]
    # read image into arrays
    mask_name = os.path.join(path_to_mask, filename)
    msk = mpimg.imread(mask_name)
    msk = msk > 127
    train_masks[i, :, :, :] = msk[:,:,None]
  print("shape of train_masks {}".format(train_masks.shape))

  return train_masks


image = get_image('/Users/liujin/Desktop/TEE_img_20')
mask = get_mask('/Users/liujin/Desktop/TEE_msk_20')

dset = h5.File('Test_20.h5', 'w')
dset.create_dataset('ih', data=image)
dset.create_dataset('b_', data=mask)


 


