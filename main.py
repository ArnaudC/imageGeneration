# https://skymind.ai/wiki/generative-adversarial-network-gan
# TODO : early stopping

import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

# Fixed parameters
mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
outputFolder = mainDir + '\\output\\'
imageHeight = 1080
imageWidth = 773
latent_dim = 100 # 100. High value means less random pictures
redimRatio = 4 # 4 min. Reduce image size : height / ratio. Dont get too low since it 'll take a huge amount of memory
GIFframeDuration = 0.5 # 0.5
dpi = 400 # 400

# Parameters that can be optimized
percentageOfImagesToKeep = 10 # 10
imagesPerIteration = 2 # Ex: 3 will generate 3x3 pictures per iteration
batch_size = 4 # 4, 32
sample_interval = 20 # 200
epochs = 401 # 30000

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
        GIFframeDuration = GIFframeDuration,
)

gan.train(
        epochs=epochs,
        batch_size=batch_size,
        sample_interval=sample_interval
)
