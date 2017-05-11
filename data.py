#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import scipy.misc
import random
import cv2

from args import Args



def normalize4gan(im):
    '''
    Convert colorspace and
    cale the input in [-1, 1] range, as described in ganhacks
    '''
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB).astype(np.float32)
    # HSV... not helpful.
    im = im.astype(np.float32)
    im /= 128.0
    im -= 1.0 # now in [-1, 1]
    return im



def denormalize4gan(im):
    '''
    Does opposite of normalize4gan:
    [-1, 1] to [0, 255].
    Warning: input im is modified in-place!
    '''
    im += 1.0 # in [0, 2]
    im *= 127.0 # in [0, 255]
    return im.astype(np.uint8)



def make_hdf5(ofname, wildcard):
    '''
    Preprocess files given by wildcard and save them in hdf5 file, as ofname.
    '''
    pool = list(glob.glob(wildcard))
    if Args.dataset_sz <= 0:
        fnames = pool
    else:
        fnames = []
        for i in range(Args.dataset_sz):
            # possible duplicate but don't care
            fnames.append(random.choice(pool))

    with h5py.File(ofname, "w") as f:
        faces = f.create_dataset("faces", (len(fnames), Args.sz, Args.sz, 3), dtype='f')

        for i, fname in enumerate(fnames):
            print(fname)
            im = scipy.misc.imread(fname, mode='RGB') # some have alpha channel
            im = scipy.misc.imresize(im, (Args.sz, Args.sz))
            faces[i] = normalize4gan(im)



def test(hdff):
    '''
    Reads in hdf file and check if pixels are scaled in [-1, 1] range.
    '''
    with h5py.File(hdff, "r") as f:
        X = f.get("faces")
        print(np.min(X[:,:,:,0]))
        print(np.max(X[:,:,:,0]))
        print(np.min(X[:,:,:,1]))
        print(np.max(X[:,:,:,1]))
        print(np.min(X[:,:,:,2]))
        print(np.max(X[:,:,:,2]))
        print("Dataset size:", len(X))
        assert np.max(X) <= 1.0
        assert np.min(X) >= -1.0



if __name__ == "__main__" :
    # Thankfully the dataset is in PNG, not JPEG.
    # Anime style suffers from significant quality degradation in JPEG.
    make_hdf5("data.hdf5", "animeface-character-dataset/thumb/*/*.png")
    #make_hdf5("data.hdf5", "animeface-character-dataset/thumb/025*/*.png")

    # Uncomment and run test, if you want.
    test("data.hdf5")
