#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
import scipy
import h5py
from args import Args
from data import denormalize4gan
#from layers import bilinear2x
from discrimination import MinibatchDiscrimination
from nets import build_discriminator, build_gen, build_enc

#import tensorflow as tf
#import keras
#keras.backend.get_session().run(tf.initialize_all_variables())



def sample_faces( faces ):
    reals = []
    for i in range( Args.batch_sz ) :
        j = random.randrange( len(faces) )
        face = faces[ j ]
        reals.append( face )
    reals = np.array(reals)
    return reals



def binary_noise(cnt):
    # Distribution of noise matters.
    # If you use single ranf that spans [0, 1], training will not work.
    # Well, for me at least.
    # Either normal or ranf works for me but be sure to use them with randrange(2) or something.
    #noise = np.random.normal( scale=Args.label_noise, size=((Args.batch_sz,) + Args.noise_shape) )

    # Note about noise rangel.
    # 0, 1 noise vs -1, 1 noise. -1, 1 seems to be better and stable.

    noise = Args.label_noise * np.random.ranf((cnt,) + Args.noise_shape) # [0, 0.1]
    noise -= 0.05 # [-0.05, 0.05]
    noise += np.random.randint(0, 2, size=((cnt,) + Args.noise_shape))

    noise -= 0.5
    noise *= 2
    return noise



def sample_fake( gen ) :
    noise = binary_noise(Args.batch_sz)
    fakes = gen.predict(noise)
    return fakes, noise



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
    #alles = scipy.misc.imresize(alles, 200) # uncomment to scale
    scipy.misc.imsave( ofname, alles )



def build_networks():
    shape = (Args.sz, Args.sz, Args.ch)

    # Learning rate is important.
    # Optimizers are important too, try experimenting them yourself to fit your dataset.
    # I recommend you read DCGAN paper.

    # Unlike gan hacks, sgd doesn't seem to work well.
    # DCGAN paper states that they used Adam for both G and D.
    #opt  = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True)
    #dopt = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True)

    # lr=0.010. Looks good, statistically (low d loss, higher g loss)
    # but too much for the G to create face.
    # If you see only one color 'flood fill' during training for about 10 batches or so,
    # training is failing. If you see only a few colors (instead of colorful noise)
    # then lr is too high for the opt and G will not have chance to form face.
    #dopt = Adam(lr=0.010, beta_1=0.5)
    #opt  = Adam(lr=0.001, beta_1=0.5)

    # vague faces @ 500
    # Still can't get higher frequency component.
    #dopt = Adam(lr=0.0010, beta_1=0.5)
    #opt  = Adam(lr=0.0001, beta_1=0.5)

    # better faces @ 500
    # but mode collapse after that, probably due to learning rate being too high.
    # opt.lr = dopt.lr / 10 works nicely. I found this with trial and error.
    # now same lr, as we are using history to train D multiple times.
    # I don't exactly understand how decay parameter in Adam works. Certainly not exponential.
    # Actually faster than exponential, when I look at the code and plot it in Excel.
    dopt = Adam(lr=0.00005, beta_1=0.5)
    opt  = Adam(lr=0.00005, beta_1=0.5)

    # too slow
    # Another thing about LR.
    # If you make it small, it will only optimize slowly.
    # LR only has to be smaller than certain threshold that is data dependent.
    # (related to the largest gradient that prevents optimization)
    #dopt = Adam(lr=0.000010, beta_1=0.5)
    #opt  = Adam(lr=0.000001, beta_1=0.5)

    # generator part
    gen = build_gen( shape )
    # loss function doesn't seem to matter for this one, as it is not directly trained
    gen.compile(optimizer=opt, loss='binary_crossentropy')
    gen.summary()

    # discriminator part
    disc = build_discriminator( shape )
    disc.compile(optimizer=dopt, loss='binary_crossentropy')
    disc.summary()

    # GAN stack
    # https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html is the faster way.
    # Here, for simplicity, I use slower way (slower due to duplicate computation).
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

    shape = (Args.sz, Args.sz, Args.ch)
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
    while epoch < 200 :
        for i in range(10) :
            reals = sample_faces( faces  )
            fakes, noises = sample_fake( gen )
            loss = autoenc.train_on_batch( reals, reals )
            epoch += 1
            print(epoch, loss)
        fakes = autoenc.predict(reals)
        dump_batch(fakes, 4, "fakes.png")
        dump_batch(reals, 4, "reals.png")
    gen.save_weights(Args.genw)
    enc.save_weights(Args.discw)
    print("Saved", Args.genw, Args.discw)



def load_weights(model, wf):
    '''
    I find error message in load_weights hard to understand sometimes.
    '''
    try:
        model.load_weights(wf)
    except:
        print("failed to load weight", wf)
        raise



def train_gan( dataf ) :
    gen, disc, gan = build_networks()

    # Uncomment these, if you want to continue training from some snapshot.
    # (or load pretrained generator weights)
    #load_weights(gen, Args.genw)
    #load_weights(disc, Args.discw)

    logger = CSVLogger('loss.csv') # yeah, you can use callbacks independently
    logger.on_train_begin() # initialize csv file
    with h5py.File( dataf, 'r' ) as f :
        faces = f.get( 'faces' )

        if Args.ch == 1:
            faces = np.array(faces[:,:,:,0])
            faces = np.expand_dims(faces, 3)
            print("xxxxxxxxxxxxxx", faces.shape)

        run_batches(gen, disc, gan, faces, logger, range(50000))
    logger.on_train_end()



def run_batches(gen, disc, gan, faces, logger, itr_generator):
    history = [] # need this to prevent G from shifting from mode to mode to trick D.
    train_disc = True
    for batch in itr_generator:
        # Using soft labels here.
        lbl_fake = Args.label_noise * np.random.ranf(Args.batch_sz)
        lbl_real = 1 - Args.label_noise * np.random.ranf(Args.batch_sz)

        fakes, noises = sample_fake( gen )
        reals = sample_faces( faces )
        # Add noise...
        # My dataset works without this.
        #reals += 0.5 * np.exp(-batch/100) * np.random.normal( size=reals.shape )

        if batch % 10 == 0 :
            if len(history) > Args.history_sz:
                history.pop(0) # evict oldest
            history.append( (reals, fakes) )

        gen.trainable = False
        #for reals, fakes in history:
        d_loss1 = disc.train_on_batch( reals, lbl_real )
        d_loss0 = disc.train_on_batch( fakes, lbl_fake )
        gen.trainable = True
       
        #if d_loss1 > 15.0 or d_loss0 > 15.0 :
        # artificial training of one of G or D based on
        # statistics is not good at all.

        # pretrain train discriminator only
        if batch < 20 :
            print( batch, "d0:{} d1:{}".format( d_loss0, d_loss1 ) )
            continue

        disc.trainable = False
        g_loss = gan.train_on_batch( noises, lbl_real ) # try to trick the classifier.
        disc.trainable = True

        # To escape this loop, both D and G should be trained so that
        # D begins to mark everything that's wrong that G has done.
        # Otherwise G will only change locally and fail to escape the minima.
        #train_disc = True if g_loss < 15 else False

        print( batch, "d0:{} d1:{}   g:{}".format( d_loss0, d_loss1, g_loss ) )

        # save weights every 10 batches
        if batch % 10 == 0 and batch != 0 :
            end_of_batch_task(batch, gen, disc, reals, fakes)
            row = {"d_loss0": d_loss0, "d_loss1": d_loss1, "g_loss": g_loss}
            logger.on_epoch_end(batch, row)



_bits = binary_noise(Args.batch_sz)
def end_of_batch_task(batch, gen, disc, reals, fakes):
    try :
        # Dump how the generator is doing.
        # Animation dump
        dump_batch(reals, 4, "reals.png")
        dump_batch(fakes, 4, "fakes.png") # to check how noisy the image is
        frame = gen.predict(_bits)
        animf = os.path.join(Args.anim_dir, "frame_{:08d}.png".format(batch))
        dump_batch(frame, 4, animf)
        dump_batch(frame, 4, "frame.png")

        serial = int(batch / 10) % 10
        prefix = os.path.join(Args.snapshot_dir, str(serial) + ".")

        print("Saving weights", serial)
        gen.save_weights(prefix + Args.genw)
        disc.save_weights(prefix + Args.discw)
    except KeyboardInterrupt :
        print("Saving, don't interrupt with Ctrl+C!", serial)
        # recursion to surely save everything haha
        end_of_batch_task(batch, gen, disc, reals, fakes)
        raise



def generate( genw, cnt ):
    shape = (Args.sz, Args.sz, Args.ch)
    gen = build_gen( shape )
    gen.compile(optimizer='sgd', loss='mse')
    load_weights(gen, Args.genw)

    generated = gen.predict(binary_noise(Args.batch_sz))
    # Unoffset, in batch.
    # Must convert back to unit8 to stop color distortion.
    generated = denormalize4gan(generated)

    for i in range(cnt):
        ofname = "{:04d}.png".format(i)
        scipy.misc.imsave( ofname, generated[i] )



def main( argv ) :
    if not os.path.exists(Args.snapshot_dir) :
        os.mkdir(Args.snapshot_dir)
    if not os.path.exists(Args.anim_dir) :
        os.mkdir(Args.anim_dir)

    # test the capability of generator network through autoencoder test.
    # The argument is that if the generator network can memorize the inputs then
    # it should be enough to GAN-generate stuff.
    # Pretraining gen isn't that useful in gan training as
    # the untrained discriminator will soon ruin everything.
    #train_autoenc( "data.hdf5" )

    # train GAN with inputs in data.hdf5
    train_gan( "data.hdf5" )

    # Lets generate stuff
    #generate( "gen.hdf5", 256 )



if __name__ == '__main__':
    main(sys.argv)
