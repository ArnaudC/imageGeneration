import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
imageHeight = 1080
imageWidth = 773
redimRatio = 8 # Reduce image size : height / ratio
percentageOfImagesToKeep = 10

(x, new_image_height, new_image_width) = loadFolderToTensorFlow(
        folder = inputPath,
        image_width = imageWidth,
        image_height = imageHeight,
        ratio = redimRatio,
        percentageOfImagesToKeep = percentageOfImagesToKeep,
        resizedFolder = resizedFolder
)
# x = None # Switch mode : digit / tcg

from GAN import GAN

gan = GAN(
        x,
        img_rows = new_image_height,
        img_cols = new_image_width,
)

gan.train(
        epochs=3001, # 30000
        batch_size=4, # 2, 32
        sample_interval=100 # 200
)
