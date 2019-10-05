# https://skymind.ai/wiki/generative-adversarial-network-gan
# TODO : tester les VAE

import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

# Fixed parameters
mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
outputFolder = mainDir + '\\output\\'
imageHeight = 1080
imageWidth = 773

# Parameters that can be optimized
redimRatio = 4 # Reduce image size : height / ratio. Dont get too low since it 'll take a huge amount of memory
percentageOfImagesToKeep = 20
imagesPerIteration = 2 # Ex: 3 will generate 3x3 pictures per iteration
latent_dim = 110 # High value means less random pictures
dpi = 200

(x, new_image_height, new_image_width) = loadFolderToTensorFlow(
        folder = inputPath,
        image_width = imageWidth,
        image_height = imageHeight,
        ratio = redimRatio,
        percentageOfImagesToKeep = percentageOfImagesToKeep,
        resizedFolder = resizedFolder,
        outputFolder = outputFolder,
)
# x = None # Switch mode : digit / tcg

from GAN import GAN

gan = GAN(
        x,
        img_rows = new_image_height,
        img_cols = new_image_width,
        outputFolder = outputFolder,
        imagesPerIteration = imagesPerIteration,
        redimRatio = redimRatio,
        percentageOfImagesToKeep = percentageOfImagesToKeep,
        latent_dim = latent_dim,
        dpi = dpi,
)

gan.train(
        epochs=1001, # 30000
        batch_size=4, # 2, 32
        sample_interval=50 # 200
)
