#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import scipy.misc
import random

from args import Args


def normalize4gan(im):
    '''
    Scale the input in [-1, 1] range, as described in ganhacks
    Warning: input im is modified in-place!
    '''
    im /= 128 # in [0, 2]
    im -= 1 # in [-1, 1]
    return im



def denormalize4gan(im):
    '''
    Does opposite of normalize4gan:
    [-1, 1] to [0, 255].
    Warning: input im is modified in-place!
    '''
    im += 1
    im *= 128
    return im



def make_hdf5(ofname, wildcard):
    '''
    Preprocess files given by wildcard and save them in hdf5 file, as ofname.
    '''
    pool = list(glob.glob(wildcard))
    fnames = []
    for i in range(Args.dataset_sz):
        # duplicate possible, but don't care.
        fnames.append(random.choice(pool))

    with h5py.File(ofname, "w") as f:
        faces = f.create_dataset("faces", (len(fnames), Args.sz, Args.sz, 3), dtype='f')

        for i, fname in enumerate(fnames):
            print(fname)
            im = scipy.misc.imread(fname, mode='RGB') # some have alpha channel
            im = scipy.misc.imresize(im, (Args.sz, Args.sz))
            im = im.astype(np.float32)

            # scale the input in [-1, 1] range, as described in ganhacks
            faces[i] = normalize4gan(im)



def test(hdff):
    '''
    Reads in hdf file and check if pixels are scaled in [-1, 1] range.
    '''
    with h5py.File(hdff, "r") as f:
        Xs = f.get("faces")
        for i in range(len(Xs)):
            X = Xs[i]
            print(X.shape)
            print(np.max(X))
            print(np.min(X))
            assert np.max(X) <= 1.0
            assert np.min(X) >= -1.0



if __name__ == "__main__" :
    # Thankfully the dataset is in PNG, not JPEG.
    # Anime style suffers from significant quality degradation in JPEG.
    make_hdf5("data.hdf5", "animeface-character-dataset/thumb/*/*.png")

    # Uncomment and run test, if you want.
    #test("data.hdf5")
