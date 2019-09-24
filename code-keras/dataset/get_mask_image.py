import h5py as h5
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.io as sio


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
  train_masks[train_masks>=1]=255
  for i in range(2):
    #print(train_masks[i])
    print(np.max(train_masks[i]))
    print(train_masks.dtype)
    im = Image.fromarray(np.squeeze(train_masks[i]), 'L')
    filename= "mask_" + str(i)
    im.save("{}.jpeg".format(filename))
  print(train_masks.shape)

  return train_masks

category = [13]
train = np.empty(shape = [0, 128, 128, 3], dtype = 'float32')
mask = np.empty(shape = [0, 128, 128, 1], dtype = 'float32')
for i in category:
  path_image = os.path.join('/Users/liujin/Desktop/Labels', str(i))
  train_masks = get_mask(i)
  mask = np.append(mask, train_masks, axis=0)


