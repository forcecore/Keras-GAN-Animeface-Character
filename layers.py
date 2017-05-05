#!/usr/bin/env python3
from keras.layers.convolutional import Conv2DTranspose
from keras.initializers import Constant
import numpy as np



def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


def bilinear2x(x, nfilters):
	'''
    Ugh, I don't like making layers.
    My credit goes to: https://kivantium.net/keras-bilinear
    '''
	return Conv2DTranspose(nfilters, (4, 4),
        strides=(2, 2),
        padding='same',
		kernel_initializer=Constant(bilinear_upsample_weights(2, nfilters)))(x)
