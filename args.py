#!/usr/bin/env python3

class Args :
    dataset_sz = 2048

    # Archive outputs of training here for animating later.
    anim_dir = "anim"

    # images size we will work on. (sz, sz, 3)
    sz = 32
    
    # alpha, used by leaky relu.
    alpha = 0.2

    # batch size, during training.
    batch_sz = 64

    # Length of the noise vector to generate the faces from.
    # Latent space z
    noise_shape = (1, 1, 128)

    # GAN training can be ruined any moment if not careful.
    # Archive some snapshots in this directory.
    snapshot_dir = "./snapshots"

    # dropout probability
    dropout = 0.3

    # noisy label magnitude
    label_noise = 0.1

    # history to keep. Slower training but higher quality.
    history_sz = 8

    genw = "gen.hdf5"
    discw = "disc.hdf5"

    # Weight initialization function.
    # This one does matter, default 'glorot_uniform' doesn't seem to work well.
    kernel_initializer = 'Orthogonal'
