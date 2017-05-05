#!/usr/bin/env python3

class Args :
    dataset_sz = 2048

    # images size we will work on. (sz, sz, 3)
    sz = 32
    
    # alpha, used by leaky relu.
    alpha = 0.2

    # batch size, during training.
    batch_sz = 32

    # Length of the noise vector to generate the faces from.
    noise_shape = (1, 1, 256)

    # GAN training can be ruined any moment if not careful.
    # Archive some snapshots in this directory.
    snapshot_dir = "./snapshots"

    # dropout probability
    dropout = 0.3

    # noisy label magnitude
    label_noise = 0.1
