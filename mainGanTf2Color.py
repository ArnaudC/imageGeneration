import os, sys
from loadFolderToTensorFlow import loadFolderToTensorFlow

# Fixed parameters
mainDir = os.path.dirname(os.path.realpath(__file__))
inputPath = mainDir + '\\input\\'
resizedFolder = mainDir + '\\resized\\'
outputFolder = mainDir + '\\output\\'
imageHeight = 1080
imageWidth = 773
channels = 3
redimRatio = 4 # 4 min. Reduce image size : height / ratio. Dont get too low since it 'll take a huge amount of memory
dpi = 100 # 400

# Parameters that can be optimized
percentageOfImagesToKeep = 1 # 10
batchSize = 1 # 4, 32
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

from GANtf2Color import GANtf2Color

gan = GANtf2Color(
    x,
    imgRows = new_image_height,
    imgCols = new_image_width,
    channels = channels,
    outputFolder = outputFolder,
    redimRatio = redimRatio,
    percentageOfImagesToKeep = percentageOfImagesToKeep,
    batchSize = batchSize,
    epochs = epochs,
    dpi = dpi,
)

gan.run(
)
