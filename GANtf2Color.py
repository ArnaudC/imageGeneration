from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import glob
import imageio# !pip install imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import IPython

from IPython import display


class GANtf2Color():
  def __init__(self, imagesInNumpyArray, imgRows, imgCols, outputFolder, channels = 1, redimRatio = 1, percentageOfImagesToKeep = 100):
    self.outputFolder = outputFolder
    self.imgRows = 28 if (imagesInNumpyArray is None) else imgRows
    self.imgCols = 28 if (imagesInNumpyArray is None) else imgCols
    self.channels = 1 if (imagesInNumpyArray is None) else channels
    self.img_shape = (self.imgRows, self.imgCols, self.channels)
    self.imagesInNumpyArray = imagesInNumpyArray
    self.redimRatio = redimRatio
    self.percentageOfImagesToKeep = percentageOfImagesToKeep

  def run(self):
    # todo : clean up this mess
    print("Tensorflow version " + tf.__version__)
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], self.imgRows, self.imgCols, self.channels).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    self.BUFFER_SIZE = 60000
    self.BATCH_SIZE = 256

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

    # The Generator
    self.generator = self.make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = self.generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    mainDir = os.path.dirname(os.path.realpath(__file__))
    outputFolder = mainDir + '\\output\\'
    fileName = "{outputFolder}gantf2.png".format(outputFolder=outputFolder,)
    dpi=100
    plt.savefig(fileName, dpi=dpi)
    plt.close()

    # The Discriminator
    self.discriminator = self.make_discriminator_model()
    decision = self.discriminator(generated_image)
    print(decision)

    # Define the loss and optimizers
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # This method returns a helper function to compute cross entropy loss

    # Optimizer
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Save checkpoints
    checkpoint_dir = './training_checkpoints'
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer,
        generator=self.generator,
    discriminator=self.discriminator)

    # Define the training loop
    self.EPOCHS = 5 # 500
    self.noise_dim = 100
    num_examples_to_generate = 16
    self.seed = tf.random.normal([num_examples_to_generate, self.noise_dim]) # We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)

    # Main call
    self.train(train_dataset, self.EPOCHS)

    # Restore the latest checkpoint.
    self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Display image
    self.display_image(self.EPOCHS)

    # Create a GIF
    self.create_gif()

  # The Generator
  def make_generator_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

  def make_discriminator_model(self):
      model = tf.keras.Sequential()
      model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.img_shape)) # test self.img_shape
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Flatten())
      model.add(layers.Dense(1))

      return model

  def discriminator_loss(self, real_output, fake_output):
    real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

  def generator_loss(self, fake_output):
    return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  def generate_and_save_images(self, model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('output/image_at_epoch_{:04d}.png'.format(epoch))
  #   plt.show()

  @tf.function
  def train_step(self, images):
    noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator(noise, training=True)

      real_output = self.discriminator(images, training=True)
      fake_output = self.discriminator(generated_images, training=True)

      gen_loss = self.generator_loss(fake_output)
      disc_loss = self.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

  def train(self, dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for image_batch in dataset:
        self.train_step(image_batch)

      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      self.generate_and_save_images(self.generator, epoch + 1, self.seed)

      # Save the model every 15 epochs
      if (epoch + 1) % 15 == 0:
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    self.generate_and_save_images(self.generator, epochs, self.seed)

  # Display a single image using the epoch number
  def display_image(self, epoch_no):
    return PIL.Image.open('output/image_at_epoch_{:04d}.png'.format(epoch_no))

  # Create the gif
  def create_gif(self):
    anim_file = 'output/dcgan.gif' # Use imageio to create an animated gif using the images saved during training.
    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('image*.png')
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
    if IPython.version_info > (6,2,0,''):
      display.Image(filename=anim_file)

