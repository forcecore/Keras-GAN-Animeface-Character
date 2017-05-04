# Keras-GAN-Animeface-Character

GAN example for Keras. Cuz MNIST is too small and there
should an example on something more realistic.


## Useful resources, before you go on

* There are great examples using MNIST already. Be sure to check them out.
    * https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
    * https://github.com/osh/KerasGAN
    * https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
* "How to Train a GAN? Tips and tricks to make GANs work" is a must read! (GAN Hacks)
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
* In principle, GAN can learn from scratch. However, it will take ages.
* Open gan.py then at the bottom, uncomment pretrain\_gen() to run generator pretraining.
    * This will attach an encoder network to the front of the generator network
      to form an auto-encoder.
    * The auto-encoder will be trained on input images.
    * Auto-encoder actually helps the main GAN training by allowing us to evaluate
      the generator part :)
* Main training step!
    * If things go well, the discriminator loss for detecting real/fake = dloss0/dloss1 should
      be around 0.1, which means it is good at telling whether the input is real or fake.
    * If learning rate is too high, one of them will get high and training fails.
    * Or it could be lack of complexity in the discriminator layer. Add more layers.
    * On the other hand, generator loss will be relatively higher than discriminator loss.
      In my case, it oscillates in range 1 to 4.
    * When I saw values around 3 to 8 then it eventually diverged.
    * If you look at loss graph at https://github.com/osh/KerasGAN,
      they had gen loss in range of 2 to 4 too!
    * GAN training is unstable, you'll need trial and error to get the hyper-pameters right
      so that the training continues in the stable, balanced zone.
    * If you see any of the loss staying > 15 (when batch size is 32) the training is screwed.
    * In case you're seeing high generator loss, it means it can't keep up with discriminator.
      You need to increase LR. (Must be slower than discriminator though)
    * But then if you have both generator and discriminator LR too high,
      you'll likely get a uniform colored output, the networks can't converge.
    * The convergence is very sensitive with LR, beware!
    * If you see all loss < 1, then discriminator learns faster and will (hopefully)
      escape that state soon.
* If you can pretrain the generator with some MSE method and then train discriminator,
  the training will be a lot faster.
  You can't really do that when you are generating stuff from noise though.
* As described in GAN Hacks, discriminator should be ahead of the generator so that
  the generator can be "guided" by the discriminator. You may need pre-training.
  To do that, copy-paste training code for discriminator and run it for about 100 batches.
* The training takes a while. For this example on Anime Face dataset, it took about 10000 batches
  to get good results.
    * Until batch 1000, I saw just some color changes with noise.
        * If you see uniform color the training is not working.
* GAN script dumps dump.png every 10 batches.
  It will show generated faces by the generator.
  If you see only one color in the generated output,
  that means the generator is tricking the discriminator with garbage and the training is failing.
  You'll need better a more complex discriminator.
* The script also dumps weights every 10 batches. Utilize them to save training time.
  Weights before diverging is preferred :)
  Uncomment load\_weights() in train\_gan().
