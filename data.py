#!/usr/bin/env python3
import glob
import h5py
import numpy as np
import scipy.misc

from args import Args


def make_hdf5(ofname, wildcard):
    # h5py data structure works best on fixed size length.
    # For that, we don't iterate on glob and instead get the file list first.
    fnames = list(glob.glob(wildcard))
    fnames.sort()

    with h5py.File(ofname, "w") as f:
        faces = f.create_dataset("faces", (len(fnames), Args.sz, Args.sz, 3), dtype='f')

        for i, fname in enumerate(fnames):
            print(fname)
            im = scipy.misc.imread(fname, mode='RGB') # some have alpha channel
            im = scipy.misc.imresize(im, (Args.sz, Args.sz))
            im = im.astype(np.float32)

            # scale the input in [-1, 1] range, as described in ganhacks
            im /= 128 # in [0, 2]
            im -= 1 # in [-1, 1]
            
            faces[i] = im


def test(hdff):
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
