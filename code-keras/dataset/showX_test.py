from liujin_utility import *
import time
import argparse

parser = argparse.ArgumentParser(description='virtual try on training')
parser.add_argument('data', metavar = 'DIR', help='path to dataset')
parser.add_argument('--path', '--path_to_sample', help='path to save generated samples')
parser.add_argument('--channel_axis', help='how many channels')
args = parser.parse_args()


# ========= Initialization =========
K.set_learning_phase(1)

nc_in = 9
nc_out = 4
ngf = 64
ndf = 64
use_lsgan = False
use_nsgan = False # non-saturating GAN


# ========== CAGAN config ==========
nc_G_inp = 8 # [x_i x_j]
nc_G_out = 5 # [m_i_g, x_i_j(RGB)]
nc_D_inp = 6 # Pos: [x_i, x_i*m_i]; Neg1: [G_out, x_j*m_j]; Neg2: [x_i, x_j*m_j]
nc_D_out = 1
gamma_i = 0.1
use_instancenorm = True
imageSize = 128
batchSize = 128 #1
lrD = 2e-4
lrG = 2e-4

train_A = load_data(args.data)
assert len(train_A)
demo_batch = minibatchAB(train_A, batchSize)
epoch, A = next(demo_batch)

_, A = demo_batch.send(10)
showX(A, args.path)



