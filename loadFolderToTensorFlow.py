import os, sys

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np

def loadFolderToTensorFlow(folder, image_width, image_height, ratio, channels):

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")


    train_files = []
    y_train = []
    i=0
    for _file in onlyfiles:
        train_files.append(_file)
        # bw2_fr_21.jpg
        start = _file.rfind('_') + 1
        end = len(_file) - 4
        imgNb = _file[start:end]
        y_train.append(int(imgNb))
        
    print("Files in train_files: %d" % len(train_files))

    # Original Dimensions
    # image_width = 773
    # image_height = 1080
    # ratio = 4 # Redimenssionner les images

    image_width = int(image_width / ratio)
    image_height = int(image_height / ratio)

    # channels = 3
    # nb_classes = 1

    dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width), dtype=np.float32)

    # i = 0
    # for _file in train_files:
    #     img = load_img(folder + "/" + _file)  # this is a PIL image
    #     img.thumbnail((image_width, image_height))
    #     # Convert to Numpy Array
    #     x = img_to_array(img)
    #     # x = x.reshape((3, 120, 160))
    #     # Normalize
    #     x = (x - 128.0) / 128.0
    #     dataset[i] = x
    #     i += 1
    #     if i % 250 == 0:
    #         print("%d images to array" % i)
    # print("All images to array!")

    return dataset

