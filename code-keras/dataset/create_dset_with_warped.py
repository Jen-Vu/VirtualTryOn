import h5py as h5
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.io as sio
from os.path import join, basename, splitext



def get_train(path):
  # get the train images into numpy arrays
  path_to_image = path
  image_dirs = os.listdir(path_to_image)
  image_dirs.sort()

  train_images = np.full((len(image_dirs), 128, 128, 3), 0, dtype='float32')
  names = []
  for i in range(len(image_dirs)):
    filename = image_dirs[i]
    name_only = splitext(basename(filename))[0]
    print(name_only)
    names.append(np.string_(name_only))

    # read image into arrays
    image_name = os.path.join(path_to_image, filename)
    img = mpimg.imread(image_name)
    img = 2*(np.float32(img)/255.0) -1
    train_images[i, :, :, :] = img
  print("shape of train_images {}".format(train_images.shape))
  return train_images, names


def get_mask(path):
  # get the train images into numpy arrays
  path_to_image = path
  image_dirs = os.listdir(path_to_image)
  image_dirs.sort()

  train_masks = np.full((len(image_dirs), 128, 128, 1), 0, dtype='float32')
  names = []
  for i in range(len(image_dirs)):
    filename = image_dirs[i]
    name_only = splitext(basename(filename))[0]
    print(name_only)
    names.append(np.string_(name_only))

    # read image into arrays
    image_name = os.path.join(path_to_image, filename)
    img = mpimg.imread(image_name)
    train_masks[i, :, :, :] = img[:,:,None]
    train_masks = np.float32(train_masks)/255.0
  print("shape of mask_images {}".format(train_masks.shape))
  return train_masks, names



train, names = get_train('/Users/liujin/Desktop/TEE_img')
mask, names_m = get_mask('/Users/liujin/Desktop/TEE_msk')
warped_mask, names_warped = get_mask('/Users/liujin/Desktop/TEE_warped')
print(names==names_m)
  # create the dataset which only contain the needed categories
dset = h5.File('TEE_1000_warped.h5', 'w')
dset.create_dataset('ih', data=train)
dset.create_dataset('b_', data=mask)
dset.create_dataset('warped', data=warped_mask)
dset.create_dataset('names', data=names)
dset.create_dataset('names_warped',data=names_warped)

with h5.File('TEE_1000_warped.h5', 'r') as f:
  names = f['names'][:]
  for i in range(len(names)):
    print(names[i])

