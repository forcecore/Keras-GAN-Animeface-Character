#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
# GAN doesn't like spare gradients (says ganhack). LeakyReLU better.
from keras.layers.advanced_activations import LeakyReLU
#import matplotlib.pyplot as plt
# matplotlib is slow for displaying output for me.
import scipy
import h5py
from args import Args
from data import denormalize4gan
from layers import bilinear2x

#import tensorflow as tf
#import keras
#keras.backend.get_session().run(tf.initialize_all_variables())



def build_discriminator( shape ) :
    def conv2d( x, filters, shape=(5, 5), **kwargs ) :
        return Conv2D( filters, shape,
            padding='same', **kwargs )( x )

    face = Input( shape=shape )
    x = face

    x = conv2d( x, 32, input_shape=shape )
    x = LeakyReLU(alpha=Args.alpha)( x )

    x = conv2d( x, 32, strides=(2, 2) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 16 x 16

    x = conv2d( x, 64, strides=(2, 2) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 8 x 8

    x = conv2d( x, 128, strides=(2, 2) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 4 x 4

    x = conv2d( x, 256, strides=(2, 2) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 2 x 2

    x = Conv2D( 512, (2, 2) )( x )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 1x1

    x = Dense( 256 )( x )
    x = LeakyReLU(alpha=Args.alpha)( x )

    x = Dense( 1, activation='sigmoid' )( x ) # 1 when "real", 0 when "fake".
    x = Flatten()( x )

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

    x = Conv2D( 256, (1, 1), activation='sigmoid' )( x )

    return models.Model( inputs=face, outputs=x )



def build_gen( shape ) :
    def deconv2d( x, filters, shape ) :
        '''
        Conv2DTransposed gives me checkerboard artifact...
        Select one of the 3.
        '''
        # Simpe Conv2DTranspose
        #x= Conv2DTranspose( filters, shape, padding='same', strides=(2, 2) )(x)

        # simple and works
        x = UpSampling2D( (2, 2) )( x )
        x = Conv2D( filters, shape, padding='same' )( x )

        # Bilinear2x... Not sure if it is without bug, not tested yet.
        # Tend to make output blurry though
        #x = bilinear2x( x, filters )
        #x = Conv2D( filters, shape, padding='same' )( x )
        return x

    noise = Input( shape=Args.noise_shape )
    x = noise
    # 1x1x256
    # noise is not useful for generating images.

    x = deconv2d( x, 256, (5, 5) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 2x2
    x = deconv2d( x, 256, (5, 5) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 4x4

    x = deconv2d( x, 256, (5, 5) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 8 x 8

    x = deconv2d( x, 128, (5, 5) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 16 x 16

    x = deconv2d( x, 64, (5, 5) )
    x = BatchNormalization()( x )
    x = LeakyReLU(alpha=Args.alpha)( x )
    # 32 x 32

    # extra layers for less noisy generation
    for i in range( 2 ) :
        x = Conv2D( 64, (5, 5), padding='same' )( x  )
        x = BatchNormalization()( x )
        x = LeakyReLU(alpha=Args.alpha)( x )

    x = Conv2D( 3, (5, 5), padding='same', activation='tanh' )( x )

    return models.Model( inputs=noise, outputs=x )



def sample_faces( faces ):
    reals = []
    for i in range( Args.batch_sz ) :
        j = random.randrange( len(faces) )
        face = faces[ j ]
        reals.append( face )
    reals = np.array(reals)
    return reals



def sample_fake( gen ):
    # Noise can't be plural but I want to put an emphasis that it has length of batch_sz

    # Distribution of noise matters.
    # If you use single ranf that spans [0, 1], training will not work.
    # Either normal or ranf works for me but be sure to use them with randrange(2) or something.
    #noises = np.random.normal( scale=0.1, size=((Args.batch_sz,) + Args.noise_shape) )
    #noises = 0.1 * np.random.ranf( size=((Args.batch_sz,) + Args.noise_shape) )
    noises = np.random.randint(0, 2, size=((Args.batch_sz,) + Args.noise_shape)).astype(np.float32)
    fakes = gen.predict(noises)
    return fakes, noises



def dump_batch(imgs, cnt, ofname):
    '''
    Merges cnt x cnt generated images into one big image.
    Use the command
    $ feh dump.png --reload 1
    to refresh image peroidically during training!
    '''
    assert Args.batch_sz >= cnt * cnt

    rows = []

    for i in range( cnt ) :
        cols = []
        for j in range(cnt*i, cnt*i+cnt):
            cols.append( imgs[j] )
        rows.append( np.concatenate(cols, axis=1) )

    alles = np.concatenate( rows, axis=0 )
    alles = denormalize4gan( alles )
    scipy.misc.imsave( ofname, alles )



def build_networks():
    shape = (Args.sz, Args.sz, 3)

    # Learning rate is important.
    # Optimizers are important too, try experimenting them yourself to fit your dataset.
    # I recommend you read DCGAN paper.
    #opt  = optimizers.SGD(lr=0.002, decay=0.0, momentum=0.0, nesterov=True)
    #dopt = optimizers.SGD(lr=0.0010, decay=0.0, momentum=0.9, nesterov=True)
    dopt = Adam(lr=0.00010, beta_1=0.5)
    opt  = Adam(lr=0.00001, beta_1=0.5)

    # generator part
    gen = build_gen( shape )
    # generator is not directly trained. Optimizer and loss doesn't matter too much.
    gen.compile(optimizer=opt, loss='mse')
    gen.summary()

    # discriminator part
    disc = build_discriminator( shape )
    disc.compile(optimizer=dopt, loss='binary_crossentropy')
    disc.summary()

    # GAN stack
    noise = Input( shape=Args.noise_shape )
    gened = gen( noise )
    result = disc( gened )
    gan = models.Model( inputs=noise, outputs=result )
    gan.compile(optimizer=opt, loss='binary_crossentropy')
    gan.summary()

    return gen, disc, gan



def train_autoenc( dataf ):
    '''
    Train an autoencoder first to see if your network is large enough.
    '''
    f = h5py.File( dataf, 'r' )
    faces = f.get( 'faces' )

    opt = Adam(lr=0.001)

    shape = (Args.sz, Args.sz, 3)
    enc = build_enc( shape )
    enc.compile(optimizer=opt, loss='mse')
    enc.summary()

    # generator part
    gen = build_gen( shape )
    # generator is not directly trained. Optimizer and loss doesn't matter too much.
    gen.compile(optimizer=opt, loss='mse')
    gen.summary()

    face = Input( shape=shape )
    vector = enc(face)
    recons = gen(vector)
    autoenc = models.Model( inputs=face, outputs=recons )
    autoenc.compile(optimizer=opt, loss='mse')

    epoch = 0
    while epoch < 1000 :
        for i in range(10) :
            reals = sample_faces( faces  )
            fakes, noises = sample_fake( gen )
            loss = autoenc.train_on_batch( reals, reals )
            epoch += 1
            print(epoch, loss)
        fakes = autoenc.predict(reals)
        dump_batch(fakes, 4, "fakes.png")
        dump_batch(reals, 4, "reals.png")



def train_gan( dataf ) :
    gen, disc, gan = build_networks()

    # Uncomment these, if you want to continue training from some snapshot.
    #gen.load_weights( Args.genw )
    #disc.load_weights( Args.discw )

    with h5py.File( dataf, 'r' ) as f :
        faces = f.get( 'faces' )
        run_batches(gen, disc, gan, faces, 20000)



def run_batches(gen, disc, gan, faces, batch_cnt):
    train_disc = True
    for batch in range(batch_cnt) :
        # Using soft labels here.
        lbl_fake = Args.label_noise * np.random.ranf(Args.batch_sz)
        lbl_real = 1 - Args.label_noise * np.random.ranf(Args.batch_sz)

        fakes, noises = sample_fake( gen )
        reals = sample_faces( faces )
        # Add noise...
        # My dataset works without this.
        #reals += np.exp(-batch/2000) * np.random.normal( size=reals.shape )

        if train_disc :
            gen.trainable = False
            d_loss1 = disc.train_on_batch( reals, lbl_real )
            d_loss0 = disc.train_on_batch( fakes, lbl_fake )
            gen.trainable = True
       
        # pretrain train discriminator only (or, make D catch up if it is behind)
        if batch < 30 or d_loss1 >= 3.0 or d_loss0 >= 3.0 :
            print( batch, "d0:{} d1:{}".format( d_loss0, d_loss1 ) )
            train_disc = True
            continue

        disc.trainable = False
        g_loss = gan.train_on_batch( noises, lbl_real ) # try to trick the classifier.
        disc.trainable = True
        train_disc = True if g_loss < 15 else False
        print( batch, "d0:{} d1:{}   g:{}".format( d_loss0, d_loss1, g_loss ) )

        # save weights every 10 batches
        if batch % 10 == 0 and batch != 0 :
            end_of_batch_task(batch, gen, disc, reals, fakes)



_bits = np.random.randint( 0, 2,
    size=((Args.batch_sz,) + Args.noise_shape)).astype(np.float32)
def end_of_batch_task(batch, gen, disc, reals, fakes):
    try :
        # Dump how the generator is doing.
        dump_batch(fakes, 4, "fakes.png")
        dump_batch(reals, 4, "reals.png") # to see if our input is sane

        # Animation dump
        frame = gen.predict(_bits)
        animf = os.path.join(Args.anim_dir, "fakes_{:05d}.png".format(batch))
        dump_batch(frame, 4, animf)

        serial = int(batch / 10) % 10
        prefix = os.path.join(Args.snapshot_dir, str(serial) + ".")

        print("Saving weights", serial)
        gen.save_weights(prefix + Args.genw)
        disc.save_weights(prefix + Args.discw)
    except KeyboardInterrupt :
        print("Saving, don't interrupt with Ctrl+C!", serial)
        # recursion to surely save everything haha
        end_of_batch_task(batch, gen, disc, reals, fakes)



def generate( genw, cnt ):
    shape = (Args.sz, Args.sz, 3)
    gen = build_gen( shape )
    gen.compile(optimizer='sgd', loss='mse')
    gen.load_weights(genw)

    noise = np.random.randint( 0, 2, size=((cnt,) + Args.noise_shape)).astype(np.float32)
    generated = gen.predict(noise)
    # Unoffset, in batch.
    # Must convert back to unit8 to stop color distortion.
    generated = denormalize4gan(generated).astype(np.uint8)

    for i in range(cnt):
        ofname = "{:04d}.png".format(i)
        scipy.misc.imsave( ofname, generated[i] )



def main( argv ) :
    if not os.path.exists(Args.snapshot_dir) :
        os.mkdir(Args.snapshot_dir)
    if not os.path.exists(Args.anim_dir) :
        os.mkdir(Args.anim_dir)

    # test capability of generator through autoencoder test.
    #train_autoenc( "data.hdf5" )

    # train GAN with inputs in data.hdf5
    train_gan( "data.hdf5" )

    # Lets generate stuff
    #generate( "gen.hdf5", 256 )



if __name__ == '__main__':
    main(sys.argv)
