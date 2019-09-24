import h5py as h5
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.io as sio


def get_train(path, cat):
  # get the train images into numpy arrays
  path_to_image = path
  image_dirs = os.listdir(path_to_image)
  image_dirs.sort()

  train_images = np.full((len(image_dirs), 128, 128, 3), 0, dtype='float32')
  txt_name = 'seq' + str(cat)
  text_file = open(txt_name, 'w')
  for i in range(len(image_dirs)):
    filename = image_dirs[i]

    # write the name in sequence into a text file
    text_file.write(filename + '\n')

    # read image into arrays
    image_name = os.path.join(path_to_image, filename)
    img = mpimg.imread(image_name)
    img = 2*(np.float32(img)/255.0) -1
    train_images[i, :, :, :] = img
  text_file.close()
  print("shape of train_images {}".format(train_images.shape))

  return train_images


def get_mask(cat):
  # get the mask data into numpy arrays
  label_path = '/Users/liujin/Desktop/Labels/Anno/language_original.mat'
  dataset = '/Users/liujin/Desktop/G2.h5'
  with h5.File(dataset, 'r') as f:
    masks = f['b_'][:]
    masks = masks.transpose(0, 3, 2, 1)
  label = sio.loadmat(label_path)
  cat_lb = (label['cate_new'] == cat).squeeze()
  train_masks = masks[cat_lb, :, :, :]
  train_masks[train_masks!=3]=0
  train_masks[train_masks==3]=1
  print("shape of train_masks {}".train_masks.shape)

  return train_masks

category = [13]
train = np.empty(shape = [0, 128, 128, 3], dtype = 'float32')
mask = np.empty(shape = [0, 128, 128, 1], dtype = 'float32')
for i in category:
  path_image = os.path.join('/Users/liujin/Desktop/Labels', str(i))
  train_images = get_train(path_image, i)
  train_masks = get_mask(i)

  train = np.append(train, train_images, axis=0)
  mask = np.append(mask, train_masks, axis=0)

  # create the dataset which only contain the needed categories
dset = h5.File('Tee.h5', 'w')
dset.create_dataset('ih', data=train)
dset.create_dataset('b_', data=mask)
