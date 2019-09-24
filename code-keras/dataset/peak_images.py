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


def get_mask(mask_path):
  dataset = 'Tee.h5'
  with h5.File(dataset, 'r') as f:
    masks = f['b_'][:]
  masks[masks==0]=0
  masks[masks==1]=255
  img = masks.astype(np.uint8)
  print(np.max(img))
  print(img.dtype)

  for i in range(masks.shape[0]):
    im = Image.fromarray(np.squeeze(img[i]), 'L')
    filename = format(i, "08d")
    img_file = os.path.join(mask_path, filename)
    im.save("{}.jpeg".format(img_file))

  return masks


def get_image(image_path):
  dataset= 'Tee.h5'
  with h5.File(dataset, 'r') as f:
    images = f['ih'][:]
    print("the shape of images is {}".format(images.shape))
    images = ((images+1.0)/2.0)*255.0
    img = images.astype(np.uint8)
  print(img.dtype)
  print(img.shape)

  for i in range(img.shape[0]):
    im = Image.fromarray(np.squeeze(img[i]), 'RGB')
    filename = format(i, "08d")
    img_file = os.path.join(image_path, filename)
    im.save("{}.jpeg".format(img_file))

  #for i in range(8):
    #im = Image.fromarray(np.squeeze(img[i]), 'RGB')
    #filename = "img_" + str(i)
    #im.save("{}.jpeg".format(filename))
  return images



train = np.empty(shape = [0, 128, 128, 3], dtype = 'float32')
mask = np.empty(shape = [0, 128, 128, 1], dtype = 'float32')

#masks = get_mask('/Users/liujin/Desktop/train_masks')
imgs = get_image('/Users/liujin/Desktop/train_images')

 


