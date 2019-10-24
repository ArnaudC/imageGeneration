import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

# Fixed parameters
mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
outputFolder = mainDir + '\\output\\'
imageHeight = 28 # 1080
imageWidth = 28 # 773
channels = 1
redimRatio = 1 # 4 min. Reduce image size : height / ratio. Dont get too low since it 'll take a huge amount of memory

# Parameters that can be optimized
percentageOfImagesToKeep = 1 # 10

(x, new_image_height, new_image_width) = loadFolderToTensorFlow(
    folder = inputPath,
    image_width = imageWidth,
    image_height = imageHeight,
    ratio = redimRatio,
    percentageOfImagesToKeep = percentageOfImagesToKeep,
    resizedFolder = resizedFolder,
    outputFolder = outputFolder,
)

from GANtf2Color import GANtf2Color

gan = GANtf2Color(
    x,
    imgRows = new_image_height,
    imgCols = new_image_width,
    channels = channels,
    outputFolder = outputFolder,
    redimRatio = redimRatio,
    percentageOfImagesToKeep = percentageOfImagesToKeep,
)

gan.run(
)
