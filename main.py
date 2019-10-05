# https://skymind.ai/wiki/generative-adversarial-network-gan
# TODO : tester les VAE

import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
outputFolder = mainDir + '\\output\\'
imageHeight = 1080
imageWidth = 773
redimRatio = 5 # Reduce image size : height / ratio
percentageOfImagesToKeep = 1

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
)

gan.train(
        epochs=30001, # 30000
        batch_size=8, # 2, 32
        sample_interval=200 # 200
)
