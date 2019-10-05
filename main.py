# Image generation
# OCR
# Classification

import os, sys

from GAN import GAN
from loadFolderToTensorFlow import loadFolderToTensorFlow

inputPath = r"C:\Users\aorus\Dropbox\Dev\DataMining\ImageGeneration\input"

image_width = 773
image_height = 1080
channels = 3 # 3 colors : R, G, B
ratio = 4 # Reduce image size : height / ratio

x = loadFolderToTensorFlow(inputPath, image_width, image_height, channels, ratio)
# x = None # Switch mode : digit / tcg

gan = GAN(
        x,
        img_rows = image_width,
        img_cols = image_height,
        channels = channels,
)

gan.train(
        epochs=3000, # 30000, 3000
        batch_size=2, # 32, 2
        sample_interval=200
)

# d_loss_real = self.discriminator.train_on_batch(imgs, valid)
# ValueError: Error when checking input: expected input_1 to have 4 dimensions, but got array with shape (2, 3, 270, 1, 193)   
# ValueError: Error when checking input: expected input_1 to have shape (270, 193, 3) but got array with shape (3, 270, 193)
