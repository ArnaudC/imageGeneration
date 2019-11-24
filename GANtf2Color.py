from __future__ import absolute_import, division, print_function, unicode_literals

import os
import PIL
import glob
import time
import IPython
import imageio # !pip install imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Dropout, Flatten, Conv2D

from IPython import display

class GANtf2Color():
  def __init__(self, imagesInNumpyArray, imgRows, imgCols, outputFolder, checkpointFolder, batchSize, epochs, channels = 3, redimRatio = 1, percentageOfImagesToKeep = 100, dpi = 100, latentDim = 100, convolutionNb = -1, colors = 256):
    self.outputFolder = outputFolder
    self.checkpointFolder = checkpointFolder
    self.imgRows = imgRows
    self.imgCols = imgCols
    self.channels = channels
    self.imgShape = (self.imgRows, self.imgCols, self.channels)
    self.imagesInNumpyArray = imagesInNumpyArray
    self.imgNb = imagesInNumpyArray.shape[0]
    self.redimRatio = redimRatio
    self.percentageOfImagesToKeep = percentageOfImagesToKeep
    self.batchSize = batchSize
    self.epochs = epochs
    self.dpi = dpi
    self.latentDim = latentDim
    self.convolutionNb = convolutionNb
    self.colors = colors

  def run(self):
    print("Tensorflow version " + tf.__version__)
    train_images = self.imagesInNumpyArray.reshape(self.imgNb, self.imgRows, self.imgCols, self.channels).astype('float32')

    # Use mnist dataset
    # self.batchSize = 60000
    # self.batchSize = 256
    # self.imgRows = 28
    # self.imgCols = 28
    # self.channels = 1
    # (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.imgNb).batch(self.batchSize)

    # The Generator
    self.generator = self.make_generator_model_2()
    noise = np.random.normal(0, 1, (1, self.latentDim)) # r * c to plot many
    generatedImage = self.generator(noise, training=False)
    genOutput = generatedImage[0, :, :, :]
    normalized = 0.5 * tf.keras.utils.normalize(genOutput, axis=-1, order=2) + 0.5 # Normalize between 0 and 1.
    plt.imshow(normalized)
    fileName = "{outputFolder}gantf2.png".format(outputFolder=self.outputFolder,)
    plt.savefig(fileName, dpi=self.dpi)
    plt.close()

    # The Discriminator
    self.discriminator = self.make_discriminator_model_2()
    decision = self.discriminator(generatedImage)
    print(decision)

    # Define the loss and optimizers
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # This method returns a helper function to compute cross entropy loss

    # Optimizer
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Save checkpoints
    self.checkpoint_prefix = os.path.join(self.checkpointFolder, "ckpt")
    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer,
        generator=self.generator,
    discriminator=self.discriminator)

    # Define the training loop
    num_examples_to_generate = 16
    self.seed = tf.random.normal([num_examples_to_generate, self.latentDim]) # We will reuse this seed overtime (so it's easier) to visualize progress in the animated GIF)

    # Main call
    self.train(train_dataset)

    # Restore the latest checkpoint.
    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpointFolder))

    # Display image
    self.display_image(self.epochs)

    # Create a GIF
    self.create_gif()

  # The Generator
  def make_generator_model(self):
    imgRows4 = int(self.imgRows / 4)
    imgCols4 = int(self.imgCols / 4)
    imgRows2 = 2 * imgRows4
    imgCols2 = 2 * imgCols4

    model = tf.keras.Sequential()
    model.add(Dense(imgRows4 * imgCols4 * (256 * self.channels), use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((imgRows4, imgCols4, 256 * self.channels)))
    assert model.output_shape == (None, imgRows4, imgCols4, 256 * self.channels) # Note: None is the batch size

    model.add(Conv2DTranspose(128 * self.channels, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, imgRows4, imgCols4, 128 * self.channels)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64 * self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, imgRows2, imgCols2, 64 * self.channels)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1 * self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, self.imgRows, self.imgCols, 1 * self.channels)

    model.summary()
    return model

  def make_generator_model_conv_nb(self):
    imgRowsConvNb = int(self.imgRows / self.convolutionNb)
    imgColsConvNb = int(self.imgCols / self.convolutionNb)

    model = tf.keras.Sequential()
    model.add(Dense(imgRowsConvNb * imgColsConvNb * 256, use_bias=False, input_shape=(self.latentDim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((imgRowsConvNb, imgColsConvNb, 256)))
    assert model.output_shape == (None, imgRowsConvNb, imgColsConvNb, 256) # Note: None is the batch size

    for i in range(1, self.convolutionNb - 2 + 1):
      colorsI = int(self.colors / (i + 1))
      stridesI = (2, 2) if (i % 2 == 0) else (1, 1)
      imgRowsI = imgRowsConvNb * i
      imgColsI = imgColsConvNb * i
      model.add(Conv2DTranspose(256, (5, 5), strides=stridesI, padding='same', use_bias=False))
      # colorsI * self.channels
      # assert model.output_shape == (None, imgRowsI, imgColsI, 256) # colorsI * self.channels
      model.add(BatchNormalization())
      model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, self.imgRows, self.imgCols, 256)

    model.summary()
    return model

  def make_generator_model_2(self):
    imgRows4 = int(self.imgRows / 4)
    imgCols4 = int(self.imgCols / 4)
    imgRows2 = 2 * imgRows4
    imgCols2 = 2 * imgCols4

    model = tf.keras.Sequential()
    model.add(Dense(imgRows4 * imgCols4 * 256, use_bias=False, input_shape=(self.latentDim,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((imgRows4, imgCols4, 256 * self.channels)))
    assert model.output_shape == (None, imgRows4, imgCols4, 256 * self.channels) # Note: None is the batch size

    model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, imgRows2, imgCols2, 256)
    model.add(BatchNormalization())
    model.add(LeakyReLU())


    model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, imgRows2, imgCols2, 256)
    model.add(BatchNormalization())
    model.add(LeakyReLU())



    model.add(Conv2DTranspose(64 * self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, imgRows2, imgCols2, 64 * self.channels)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1 * self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, self.imgRows, self.imgCols, 1 * self.channels)

    model.summary()
    return model

  def make_discriminator_model(self):
      model = tf.keras.Sequential()
      model.add(Conv2D(64 * self.channels, (5, 5), strides=(2, 2), padding='same', input_shape=self.imgShape)) # self.generator.output_shape[1:]
      model.add(LeakyReLU())
      model.add(Dropout(0.3))

      model.add(Conv2D(128 * self.channels, (5, 5), strides=(2, 2), padding='same'))
      model.add(LeakyReLU())
      model.add(Dropout(0.3))

      model.add(Flatten())
      model.add(Dense(1))

      # model.add(layers.Activation('sigmoid')) # to test

      model.summary()
      return model

  def make_discriminator_model_conv_nb(self):
      model = tf.keras.Sequential()
      colorsN = int(self.colors / self.convolutionNb)
      model.add(Conv2D(colorsN * self.channels, (5, 5), strides=(2, 2), padding='same', input_shape=self.imgShape)) # self.generator.output_shape[1:]
      model.add(LeakyReLU())
      model.add(Dropout(0.3))

      for i in range(1, self.convolutionNb - 3 + 1):
        colorsI = int(self.colors / (i+1))
        model.add(Conv2D(colorsI * self.channels, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

      model.add(Flatten())
      model.add(Dense(1))

      model.summary()
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

    predictions = tf.keras.utils.normalize(predictions, axis=-1, order=2) * 0.5 + 0.5  # Normalize between 0 and 1.

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :]) # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
      plt.savefig('output/image_at_epoch_{:04d}.png'.format(epoch))

  @tf.function
  def train_step(self, images):
    noise = tf.random.normal([self.batchSize, self.latentDim])

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

  def train(self, dataset):
    for epoch in range(self.epochs):
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
    self.generate_and_save_images(self.generator, self.epochs, self.seed)

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

