# Keras-GAN-Animeface-Character

GAN example for Keras. Cuz MNIST is too small and there
should an example on something more realistic.


## Useful resources, before you go on

* There are great examples using MNIST already. Be sure to check them out.
    * https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
    * https://github.com/osh/KerasGAN
    * https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
* "How to Train a GAN? Tips and tricks to make GANs work" is a must read!
    * The advice was very helpful in making this example.
    * https://github.com/soumith/ganhacks


## How to run this example

* My environment: Python 3.6 + Keras 2.0.0
* I HATE making a program that has so many command line parameters to pass.
  Many of the parameters are there in the scripts. Adjust the script as you need.
  The "main()" function is at the bottom of the script as people do in C/C++
* Some global parameters are defined in args.py.
    * They are defined as class variables not instance variables so you may have trouble
      running/training multiple instances of the GAN with different parameters.
      (which is very unlikely to happen)
* Download dataset from http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
    * Extract it to this directory so that the scipt can find
      ./animeface-character-dataset/thumb/
    * Any dataset should work in principle but GAN is sensitive to hyperparameters and may not work
      on yours. I tuned the parameters for animeface-character-dataset.
* Run the preprocessing script. It saves training time to resize/scale the input than
  doing those tasks on the fly in the training loop.
    * ./data.py
    * data.py will resize the input to 64x64 (size defined in args.py) and dump them in data.hdf5.
    * Again, which files to read is defined in the script at the bottom, not by sys.argv.
