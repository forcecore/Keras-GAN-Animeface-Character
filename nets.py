#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
# GAN doesn't like spare gradients (says ganhack). LeakyReLU better.
from keras.layers.advanced_activations import LeakyReLU
#import matplotlib.pyplot as plt
# matplotlib is slow for displaying output for me.
import scipy
import h5py
from args import Args
from data import denormalize4gan
from layers import bilinear2x
from discrimination import MinibatchDiscrimination

#import tensorflow as tf
#import keras
#keras.backend.get_session().run(tf.initialize_all_variables())



def build_discriminator( shape ) :
    def conv2d( x, filters, shape=(5, 5), **kwargs ) :
        return Conv2D( filters, shape,
            padding='same', **kwargs )( x )

    # https://github.com/tdrussell/IllustrationGAN
    # As proposed by them, unlike GAN hacks, MaxPooling works better for anime dataset it seems.

    face = Input( shape=shape )
    x = face

    #x = conv2d( x, 32 )
    #x = MaxPooling2D()( x )
    ##x = BatchNormalization()( x )
    #x = LeakyReLU(alpha=Args.alpha)( x )
    # 32x32

    x = conv2d( x, 32 )
    x = MaxPooling2D()( x )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 16 x 16

    x = conv2d( x, 64 )
    x = MaxPooling2D()( x )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 8 x 8

    x = conv2d( x, 128 )
    x = MaxPooling2D()( x )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 4 x 4

    x = conv2d( x, 256 )
    x = MaxPooling2D()( x )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 2 x 2

    x = Conv2D( 512, (2, 2) )( x )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 1x1

    x = Flatten()(x)

    # add 16 features. Run 1D conv of size 3.
    x = MinibatchDiscrimination(16, 3)( x )

    x = Dense( 1, activation='sigmoid' )( x ) # 1 when "real", 0 when "fake".

    return models.Model( inputs=face, outputs=x )



def build_enc( shape ) :
    def conv2d( x, filters, shape=(5, 5), **kwargs ) :
        return Conv2D( filters, shape, padding='same', **kwargs )( x )

    face = Input( shape=shape )

    x = conv2d( face, 32, input_shape=shape )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 32 x 32
    x = conv2d( x, 32, strides=(2, 2) )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 16 x 16

    x = conv2d( x, 64, strides=(2, 2) )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 8 x 8

    x = conv2d( x, 128, strides=(2, 2) )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 4 x 4

    x = Conv2D( 256, (3, 3) )( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 2 x 2

    x = Conv2D( 256, (2, 2) )( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 1x1

    x = Conv2D( Args.noise_shape[2], (1, 1), activation='tanh' )( x )

    return models.Model( inputs=face, outputs=x )



def build_gen( shape ) :
    def deconv2d( x, filters, shape ) :
        '''
        Conv2DTransposed gives me checkerboard artifact...
        Select one of the 3.
        '''
        # Simpe Conv2DTranspose
        # Not good, compared to upsample + conv2d below.
        x= Conv2DTranspose( filters, shape, padding='same', strides=(2, 2) )(x)

        # simple and works
        #x = UpSampling2D( (2, 2) )( x )
        #x = Conv2D( filters, shape, padding='same' )( x )

        # Bilinear2x... Not sure if it is without bug, not tested yet.
        # Tend to make output blurry though
        #x = bilinear2x( x, filters )
        #x = Conv2D( filters, shape, padding='same' )( x )

        return x

    # https://github.com/tdrussell/IllustrationGAN  z predictor...?
    # might help. Not sure.

    noise = Input( shape=Args.noise_shape )
    x = noise
    # 1x1x256
    # noise is not useful for generating images.

    x = deconv2d( x, 256, (5, 5) )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 2x2
    x = deconv2d( x, 256, (5, 5) )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 4x4

    x = deconv2d( x, 256, (5, 5) )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 8 x 8

    x = deconv2d( x, 128, (5, 5) )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 16 x 16

    x = deconv2d( x, 64, (5, 5) )
    #x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 32 x 32

    #x = deconv2d( x, 64, (5, 5) )
    #x = BatchNormalization()( x )
    #x = LeakyReLU(alpha=Args.alpha)( x )
    ## 64 x 64

    x = Conv2D( 3, (5, 5), padding='same', activation='tanh' )( x )

    return models.Model( inputs=noise, outputs=x )
