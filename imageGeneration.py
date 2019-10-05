# Image generation
# OCR
# Classification

import os, sys

from GAN import GAN
from loadFolderToTensorFlow import loadFolderToTensorFlow

inputPath = r"C:\Users\aorus\Dropbox\Dev\DataMining\ImageGeneration\ctg"

x = loadFolderToTensorFlow(inputPath, image_width = 773, image_height = 1080, ratio = 4, channels = 3)
# x = None # Switch mode : digit / tcg

gan = GAN(
        x,
        channels = 3,
        img_rows = 270,
        img_cols = 193,
)

gan.train(
        epochs=3000, # 30000, 3000
        batch_size=2, # 32, 2
        sample_interval=200
)

# d_loss_real = self.discriminator.train_on_batch(imgs, valid)
# ValueError: Error when checking input: expected input_1 to have 4 dimensions, but got array with shape (2, 3, 270, 1, 193)   
# ValueError: Error when checking input: expected input_1 to have shape (270, 193, 3) but got array with shape (3, 270, 193)
