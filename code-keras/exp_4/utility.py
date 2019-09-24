from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, sigmoid
from keras.initializers import RandomNormal

from instance_normalization import InstanceNormalization

from keras.applications import *
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam
from PIL import Image
import numpy as np
from random import randint, shuffle
import h5py as h5
import scipy.misc
import os

channel_axis = -1
channel_first = False
use_instancenorm = True
use_lsgan = False
use_nsgan = False  # non-saturating GAN
isRGB = True


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a)  # for convolution kernel
    k.conv_weight = True
    return k


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)  # for batch normalization


# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                              gamma_initializer=gamma_init)


def instance_norm():
    return InstanceNormalization(axis=channel_axis, epsilon=1.01e-5,
                                 gamma_initializer=gamma_init)


def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    if channel_first:
        input_a = Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name='final'.format(out_feat, 1),
               activation="sigmoid" if use_sigmoid else None)(_)
    return Model(inputs=[input_a], outputs=_)


def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True, use_batchnorm=True):
    s = isize if fixed_input_size else None
    _ = inputs = Input(shape=(s, s, nc_in))
    x_i = Lambda(lambda x: x[:, :, :, 0:3], name='x_i')(inputs)
    y_i = Lambda(lambda x: x[:, :, :, 4:7], name='y_j')(inputs)
    xi_and_y_i = concatenate([x_i, y_i], name='xi_yi')
    xi_yi_sz64 = AveragePooling2D(pool_size=2)(xi_and_y_i)
    xi_yi_sz32 = AveragePooling2D(pool_size=4)(xi_and_y_i)
    xi_yi_sz16 = AveragePooling2D(pool_size=8)(xi_and_y_i)
    xi_yi_sz8 = AveragePooling2D(pool_size=16)(xi_and_y_i)
    layer1 = conv2d(64, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer1')(_)
    layer1 = LeakyReLU(alpha=0.2)(layer1)
    layer1 = concatenate([layer1, xi_yi_sz64])  # ==========
    layer2 = conv2d(128, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer2')(layer1)
    if use_instancenorm:
        layer2 = instance_norm()(layer2, training=1)
    else:
        layer2 = batchnorm()(layer2, training=1)
    layer3 = LeakyReLU(alpha=0.2)(layer2)
    layer3 = concatenate([layer3, xi_yi_sz32])  # ==========
    layer3 = conv2d(256, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer3')(layer3)
    if use_instancenorm:
        layer3 = instance_norm()(layer3, training=1)
    else:
        layer3 = batchnorm()(layer3, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer3)
    layer4 = concatenate([layer4, xi_yi_sz16])  # ==========
    layer4 = conv2d(512, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                    padding="same", name='layer4')(layer4)
    if use_instancenorm:
        layer4 = instance_norm()(layer4, training=1)
    else:
        layer4 = batchnorm()(layer4, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer4)
    layer4 = concatenate([layer4, xi_yi_sz8])  # ==========

    layer9 = Conv2DTranspose(256, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                             kernel_initializer=conv_init, name='layer9')(layer4)
    layer9 = Cropping2D(((1, 1), (1, 1)))(layer9)
    if use_instancenorm:
        layer9 = instance_norm()(layer9, training=1)
    else:
        layer9 = batchnorm()(layer9, training=1)
    layer9 = Concatenate(axis=channel_axis)([layer9, layer3])
    layer9 = Activation('relu')(layer9)
    layer9 = concatenate([layer9, xi_yi_sz16])  # ==========
    layer10 = Conv2DTranspose(128, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer10')(layer9)
    layer10 = Cropping2D(((1, 1), (1, 1)))(layer10)
    if use_instancenorm:
        layer10 = instance_norm()(layer10, training=1)
    else:
        layer10 = batchnorm()(layer10, training=1)
    layer10 = Concatenate(axis=channel_axis)([layer10, layer2])
    layer10 = Activation('relu')(layer10)
    layer10 = concatenate([layer10, xi_yi_sz32])  # ==========
    layer11 = Conv2DTranspose(64, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer11')(layer10)
    layer11 = Cropping2D(((1, 1), (1, 1)))(layer11)
    if use_instancenorm:
        layer11 = instance_norm()(layer11, training=1)
    else:
        layer11 = batchnorm()(layer11, training=1)
    layer11 = Activation('relu')(layer11)

    layer12 = concatenate([layer11, xi_yi_sz64])  # ==========
    layer12 = Activation('relu')(layer12)
    layer12 = Conv2DTranspose(32, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                              kernel_initializer=conv_init, name='layer12')(layer12)
    layer12 = Cropping2D(((1, 1), (1, 1)))(layer12)
    if use_instancenorm:
        layer12 = instance_norm()(layer12, training=1)
    else:
        layer12 = batchnorm()(layer12, training=1)

    layer12 = conv2d(5, kernel_size=4, strides=1, use_bias=(not (use_batchnorm and s > 2)),
                     padding="same", name='out128')(layer12)

    im_i_j = Lambda(lambda x: x[:, :, :, 0:3], name='im_i_j')(layer12)
    m_g_i = Lambda(lambda x: x[:, :, :, 3:4], name='mask_i')(layer12)
    m_g_j = Lambda(lambda x: x[:, :, :, 4:], name='mask_j')(layer12)
    
    im_i_j = Activation("tanh", name='im_i_j_tanh')(im_i_j)
    m_g_i = Activation("sigmoid", name='m_g_i_sigmoid')(m_g_i)
    m_g_j = Activation("sigmoid", name='m_g_j_sigmoid')(m_g_j)
    #encoder_outputs = Dense(units=latent_vector_len, activation=k.layers.Lambda(lambda z: k.backend.round(k.layers.activations.sigmoid(x=z))), kernel_initializer="lecun_normal")(x)

    out = concatenate([im_i_j, m_g_i, m_g_j], name='out128_concat')

    return Model(inputs=inputs, outputs=[out])


def loss_fn(output, target):
    if use_lsgan:
        return K.mean(K.abs(K.square(output-target)))
    else:
        return -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))


def loss_fn_mask(output, target):
    return K.mean(K.abs(output - target))


def cycle_variables(netG1):

    real_input = netG1.inputs[0]
    fake_out = netG1.outputs[0]
    # Legacy: how to split channels
    # https://github.com/fchollet/keras/issues/5474
    
    """                """
    """ drawing graph  """
    """                """
    
    ############################### 1st cycle ##################################
    # get the original input and the image
    x_i = Lambda(lambda x: x[:, :, :, 0:4])(real_input)
    im_i = Lambda(lambda x: x[:, :, :, 0:3])(real_input)
    warped_mask = Lambda(lambda x: x[:, :, :, 8:])(real_input)
    
    # get the output of the first generation
    im_i_j = Lambda(lambda x: x[:, :, :, 0:3])(fake_out)
    m_g_i = Lambda(lambda x: x[:, :, :, 3:4])(fake_out)
    m_g_j = Lambda(lambda x: x[:, :, :, 4:])(fake_out)

    # ========================= binarize mask ==================================
    #m_g_i = tf.to_int32(m_g_i > 0.5)
    #m_g_j = tf.to_int32(m_g_j > 0.5)
    #m_g_i = tf.cast(m_g_i, tf.float32)
    #m_g_j = tf.cast(m_g_j, tf.float32)
    #m_g_i = tf.clip_by_value(m_g_i, 0.5, 1.0)
    #print(m_g_i)
    #m_g_j = tf.clip_by_value(m_g_j, tf.constant(0.5), tf.constant(1.0))

    #mask_w = K.repeat_elements(warped_mask, rep=3, axis = -1)
 
    fake_im = warped_mask*im_i_j + (1 - warped_mask)*im_i
    # get the final fake image and mask for 1st cycle
    fake_output = concatenate([fake_im, m_g_i], axis = -1)

    ############################### 2nd cycle ##################################
    # get the input for 2nd cycle
    concat_input_G2 = concatenate([fake_output, x_i, m_g_i], axis=-1)  # swap
    
    # get the output for the 2nd cycle
    rec_output = netG1([concat_input_G2])
    rec_im_i_j = Lambda(lambda x: x[:, :, :, 0:3])(rec_output)
    rec_m_g_i = Lambda(lambda x: x[:, :, :, 3:4])(rec_output)
    rec_m_g_j = Lambda(lambda x: x[:, :, :, 4:])(rec_output)

    # ========================= binarize mask ==================================
    #rec_m_g_i = tf.to_int32(rec_m_g_i > 0.5)
    #rec_m_g_j = tf.to_int32(rec_m_g_j > 0.5)
    #rec_m_g_i = tf.cast(rec_m_g_i, tf.float32)
    #rec_m_g_j = tf.cast(rec_m_g_j, tf.float32)
    #rec_m_g_i = tf.clip_by_value(rec_m_g_i, tf.constant(0.5), tf.constant(1.0))
    #rec_m_g_j = tf.clip_by_value(rec_m_g_j, tf.constant(0.5), tf.constant(1.0))
    # calculate the fake image for second cycle

    rec_im = rec_m_g_i * rec_im_i_j + (1 - rec_m_g_i) * fake_im
    
    # get the final fake image and mask for 2nd cycle
    rec_output = concatenate([rec_im, rec_m_g_i], axis = -1)

    fn_generate = K.function([real_input], [fake_output, rec_output])
    return real_input, fake_out, rec_output, fn_generate, m_g_i, m_g_j


def D_loss(netD, real, fake, rec):

    # the input images and masks
    im_i = Lambda(lambda x: x[:, :, :, 0:3])(real)
    m_i = Lambda(lambda x: x[:, :, :, 3:4])(real)
    im_j = Lambda(lambda x: x[:, :, :, 4:7])(real)
    m_j = Lambda(lambda x: x[:, :, :, 7:8])(real)

    # the output of the 1st cycle
    im_i_j = Lambda(lambda x: x[:, :, :, 0:3])(fake)
    m_g_i = Lambda(lambda x: x[:, :, :, 3:4])(fake)
    m_g_j = Lambda(lambda x: x[:, :, :, 4:])(fake)

    # the output of the 2nd cycle
    rec_im = Lambda(lambda x: x[:, :, :, 0:3])(rec)

    # three type of predictions for Discriminator 
    pred_1 = netD(concatenate([im_i, im_i*m_i], axis = -1))  # positive
    pred_2 = netD(concatenate([im_i_j, im_j*m_j], axis = -1))  # negative, fake image
    pred_3 = netD(concatenate([im_i, im_j*m_j], axis = -1))  # negative, image and clothes mismatch

    # losses for the three types
    loss_1 = loss_fn(pred_1, K.ones_like(pred_1))
    loss_2 = loss_fn(pred_2, K.zeros_like(pred_2))
    loss_3 = loss_fn(pred_3, K.zeros_like(pred_3)) 

    # loss for 1st generation (including loss for masks)
    loss_G = loss_fn(pred_2, K.ones_like(pred_2)) + loss_fn_mask(m_g_i, m_i) + loss_fn_mask(m_g_j, m_j)

    # loss for discriminator
    loss_D = loss_1 + loss_2 + loss_3

    # cycle loss 
    loss_cyc = K.mean(K.abs(rec_im - im_i)) 

    return loss_D, loss_G, loss_cyc


def load_data(dataset):

    with h5.File(dataset, 'r') as f:
        images = f['ih'][:]
        masks = f['b_'][:]
        warped = f['warped'][:]
        names = f['names'][:]
        names_warped = f['names_warped'][:]

        # concatenate images and masks
        data = np.concatenate([images, masks], -1)

        return data, warped, names, names_warped


def read_image(data, warped, names, names_warped, idx_i):

    length = len(data)

    # Load consumer picture
    input_i = data[idx_i,:,:,:]
    print("dimension of input_i {}".format(input_i.shape))

    # Load model picture randomly
    idx_j = np.random.choice(length)
    while idx_j == idx_i:
        idx_j = np.random.choice(length)
    input_j = data[idx_j,:,:,:]

    # get the warped mask
    i_name = names[idx_i]
    j_name = names[idx_j]
    i_j_name = i_name + j_name

    names_warped = names_warped.tolist()
    ind = names_warped.index(i_j_name)
    warped_mask = warped[ind, :, :, :]
    print("dimension of warped_mask {}".format(warped_mask.shape))

    input_ = np.concatenate([input_i, input_j, warped_mask], axis=-1)
    assert input_.shape[-1] == 9

    return input_


def minibatch(data, warped, names, names_warped, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    np.random.shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            np.random.shuffle(data)
            i = 0
            epoch+=1
        rtn = [read_image(data, warped, names, names_warped, j) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, warped, names, names_warped, batchsize):
    batchA=minibatch(dataA, warped, names, names_warped, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A


def showX(X, path_to_save):
    length = len(X)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    for i in range(length):
        # save consumer image
        consumer_im = X[i,:,:,:3]
        save_images('{}/consumer_{}.png'.format(path_to_save, str(i)), consumer_im)
        # save model image
        model_im = X[i,:,:,4:7]
        save_images('{}/model_{}.png'.format(path_to_save, str(i)), model_im)
        

def showG(cycleA_generate, A, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    length = len(A)
    rA = cycleA_generate([A])
    fake_output = rA[0]
    for i in range(length):
        fake_im = fake_output[i,:,:,1:]
        save_images('{}/fake_{}.png'.format(path_to_save, str(i)), fake_im)


def save_images(path, image):
    return scipy.misc.imsave(path, image)

