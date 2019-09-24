import h5py as h5
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import scipy.io as sio
import scipy


def save_images(path, image):
    return scipy.misc.imsave(path, image)


def show_mask(dataset, mask_path):
  with h5.File(dataset, 'r') as f:
    masks = f['b_'][:]
  masks[masks==0]=0
  masks[masks==1]=255
  img = masks.astype(np.uint8)
  print(np.max(img))
  print(img.dtype)

  for i in range(10):
    im = Image.fromarray(np.squeeze(img[i]), 'L')
    filename = format(i, "08d")
    img_file = os.path.join(mask_path, filename)
    im.save("{}.jpeg".format(img_file))



def show_image(dataset, image_path):
  with h5.File(dataset, 'r') as f:
    images = f['ih'][:]
    print("the shape of images is {}".format(images.shape))
    images = ((images+1.0)/2.0)*255.0
    img = images.astype(np.uint8)
  print(img.dtype)
  print(img.shape)

  for i in range(10):
    im = Image.fromarray(np.squeeze(img[i]), 'RGB')
    filename = format(i, "08d")
    img_file = os.path.join(image_path, filename)
    im.save("{}.jpeg".format(img_file))


def show_mask_warped(dataset, mask_path):
  with h5.File(dataset, 'r') as f:
    masks = f['warped'][:]
  masks[masks==0]=0
  masks[masks==1]=255
  img = masks.astype(np.uint8)
  print(np.max(img))
  print(img.dtype)

  for i in range(10):
    im = Image.fromarray(np.squeeze(img[i]), 'L')
    filename = format(i, "08d")
    img_file = os.path.join(mask_path, filename)
    im.save("{}.jpeg".format(img_file))


train = np.empty(shape = [0, 128, 128, 3], dtype = 'float32')
mask = np.empty(shape = [0, 128, 128, 1], dtype = 'float32')


show_image('TEE_1000_warped.h5', 'image')
show_mask('/Users/liujin/Desktop/exp_4/TEE_1000_warped.h5', 'mask')
show_mask_warped('TEE_1000_warped.h5','warped')


