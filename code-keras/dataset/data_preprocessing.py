import h5py as h5
import numpy as np
from matplotlib import pyplot as plt

file_1347 = '1347.h5'
file_G2 = '/Users/liujin/Desktop/G2.h5'
file_one_cat = 'one_cat.h5'
file_Tee_1000 = 'Tee_1000.h5'


def PEAK(filename, peak_number):
    with h5.File(filename, 'r') as f:
        train_images = f['ih'][:100]
        peak_image = train_images[peak_number]
        plt.imshow(peak_image)
        plt.show()

        masks = f['b_'][:100]
        peak_mask = masks[peak_number].reshape(128,128)
        plt.imshow(peak_mask, cmap='gray')
        plt.show()
    return peak_number

peak_id = PEAK(file_Tee_1000, 70)


def check_data(filename):
  with h5.File(filename, 'r') as f: 
    train_sample = f['ih'][100:1000]
    print('data type of array is {}'.format(train_sample.dtype))
    print('max value of pixel is {}'.format(np.amax(train_sample)) + '\n')
    print('min value of pixel is {}'.format(np.amin(train_sample)) + '\n')


check_data(file_1347)
check_data(file_G2) 
check_data(file_one_cat)
