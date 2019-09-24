from utility import *
import time
import argparse

parser = argparse.ArgumentParser(description='virtual try on training')
parser.add_argument('data', metavar = 'DIR', help='path to dataset')
parser.add_argument('--path', '--path_to_sample', default='examples', help='path to save generated samples')
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

########################## define NN model #############################
netGA = UNET_G(imageSize, nc_G_inp, nc_G_out, ngf)
#netGA.summary()
netDA = BASIC_D(nc_D_inp, ndf, use_sigmoid = not use_lsgan)
#netDA.summary()

########################## get the variables ###########################
real_A, fake_B, rec_A, cycleA_generate, m_g_i, m_g_j = cycle_variables(netGA)
loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_B, rec_A)

########################## define losses ###############################
loss_cyc = loss_cycA
loss_id = K.mean(K.abs(m_g_i)) +  K.mean(K.abs(m_g_j)) # loss of alpha
loss_G = loss_GA + 1*(1*loss_cyc + gamma_i*loss_id)
loss_D = loss_DA*2

########################## set training ################################
weightsD = netDA.trainable_weights
weightsG = netGA.trainable_weights
# define discriminator training function 
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD,[],loss_D)
netD_train = K.function([real_A], [loss_DA/2], training_updates)
# define generator training function 
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG,[],loss_G)
netG_train = K.function([real_A], [loss_GA, loss_cyc], training_updates)



############################### training process ########################
# load training data
train_A = load_data(args.data)
assert len(train_A)
t0 = time.time()
niter = 10000
gen_iterations = 0
epoch = 0
errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0
display_iters = 50
train_batch = minibatchAB(train_A, batchSize)


while gen_iterations < niter:

    epoch, A = next(train_batch)
    print(A.shape)

    errDA  = netD_train([A])
    errDA_sum +=errDA[0]
    errGA, errCyc = netG_train([A])
    errGA_sum += errGA
    errCyc_sum += errCyc

    gen_iterations+=1
    if gen_iterations%display_iters==0:
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_cyc: %f'
              % (epoch, niter, gen_iterations, errDA_sum/display_iters,
                 errGA_sum/display_iters, errCyc_sum/display_iters), time.time()-t0)
        errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0


######################## demo ############################################
demo_batch = minibatchAB(train_A, batchSize)
epoch, A = next(demo_batch)

_, A = demo_batch.send(20)
showX(A, args.path)
showG(cycleA_generate, A, args.path)


