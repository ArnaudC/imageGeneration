# https://skymind.ai/wiki/generative-adversarial-network-gan
from __future__ import print_function, division

import tensorflow as tf
from shutil import copyfile
from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import os, sys
import matplotlib.pyplot as plt
import sys
import numpy as np
import imageio # !pip install -q imageio
import glob
import time
import imageio

class GAN():
    def __init__(self, imagesInNumpyArray, img_rows, img_cols, outputFolder, redimRatio = 1, percentageOfImagesToKeep  = 100, imagesPerIteration = 5, channels = 3, latent_dim = 100, dpi = 100, GIFframeDuration=1):
        self.outputFolder = outputFolder
        self.img_rows = 28 if (imagesInNumpyArray is None) else img_rows
        self.img_cols = 28 if (imagesInNumpyArray is None) else img_cols
        self.channels = 1 if (imagesInNumpyArray is None) else channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.imagesInNumpyArray = imagesInNumpyArray
        self.imagesPerIteration = imagesPerIteration
        self.redimRatio = redimRatio
        self.percentageOfImagesToKeep = percentageOfImagesToKeep
        self.dpi = dpi
        self.GIFframeDuration = GIFframeDuration

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model (stacked generator and discriminator). Trains the generator to fool the discriminator.
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512)) # 512
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024)) # 1024
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        # model.add(Dense(units=1024)) # Test : new layer
        # model.add(LeakyReLU(0.1))
        # model.add(Dropout(0.2))
        model.add(Dense(units=512)) # 512
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(0.3)) # Test : Dropout
        model.add(Dense(units=256)) # 256
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(0.1)) # Test : Dropout
        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        start = time.time()

        # Load the dataset
        if (self.imagesInNumpyArray is None):
            (X_train, _), (_, _) = mnist.load_data()
        else:
            X_train = self.imagesInNumpyArray

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        if (self.imagesInNumpyArray is None):
            X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size) # Select a random batch of images
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise) # Generate a batch of new images
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)) 
            if epoch % sample_interval == 0: # If at save interval => save generated image samples
                self.saveOutput(epoch)
        self.saveGif()
        print ('{} epochs in {} sec.'.format(epoch, time.time()-start))

    def saveOutput(self, epoch):
        fileName = "{outputFolder}{dpi}_dpi{pToKeep}_pKeep{redimRatio}_redimRatio_{epoch}epoch.png".format(
            outputFolder=self.outputFolder,
            dpi=self.dpi,
            pToKeep=self.percentageOfImagesToKeep,
            redimRatio=self.redimRatio,
            epoch=epoch
        )
        if (self.imagesPerIteration == 1):
            self.saveSingleOutput(fileName)
        else:
            self.saveMultipleOutput(fileName)
        copyfile(fileName, "{outputFolder}{outputFile}".format(outputFolder=self.outputFolder, outputFile="output.png"))

    def saveMultipleOutput(self, fileName):
        r, c = self.imagesPerIteration, self.imagesPerIteration
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale images 0 - 1

        # fig, axs = plt.subplots(r, c, figsize=(self.img_rows / self.imagesPerIteration, self.img_cols / self.imagesPerIteration))
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, :]) # imshow(gen_imgs[cnt, :, :,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        # fig.set_size_inches(18.5, 10.5)
        fig.savefig(fileName, dpi=self.dpi)
        plt.close()

    def saveSingleOutput(self, fileName):
        noise = np.random.normal(loc=0, scale=1, size=[1, self.latent_dim])
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5 # Rescale images 0 - 1

        plt.imshow(gen_imgs[0], interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(fileName, dpi=self.dpi)
        plt.close()

    def saveGif(self):
        fileName = "{outputFolder}{dpi}_dpi{pToKeep}_pKeep{redimRatio}_redimRatio.gif".format(
            outputFolder=self.outputFolder,
            dpi=self.dpi,
            pToKeep=self.percentageOfImagesToKeep,
            redimRatio=self.redimRatio,
        )
        images = []
        filenames = glob.glob("{outputFolder}*epoch.png".format(outputFolder=self.outputFolder,))
        filenames.sort(key=os.path.getmtime)
        for filename in filenames[1:]: # Skip first : only noise
            images.append(imageio.imread(filename))
        imageio.mimsave(fileName, images, duration=self.GIFframeDuration)
