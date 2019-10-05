import os, sys

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np
from cv2 import cv2 # Need to install an older version on windows : 'pip install opencv-python=3.3.0.9'

# Expect (self.img_rows, self.img_cols, self.channels)
# (self.img_rows, self.img_cols, self.channels) if (imagesInNumpyArray is None) else (self.channels, self.img_rows, self.img_cols)
def loadFolderToTensorFlow(folder, image_width, image_height, channels, ratio):

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")


    train_files = []
    y_train = []
    # i=0
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

    new_image_height = int(image_height / ratio)
    new_image_width = int(image_width / ratio)

    # channels = 3
    # nb_classes = 1

    dataset = np.ndarray(shape=(len(train_files), new_image_height, new_image_width, channels), dtype=np.float32)

    i = 0
    for _file in train_files:
        img = cv2.imread(folder + "/" + _file)

        # Display the image
        # img.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img_new = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        x = cv2.resize(img, dsize=(new_image_width, new_image_height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(r'C:\Users\aorus\Dropbox\Dev\DataMining\ImageGeneration\resized\\' + _file, x) # See resized images

        # img = load_img(folder + "/" + _file)  # this is a PIL image
        # img.thumbnail((new_image_width, new_image_height))
        # x = img_to_array(img) # Convert to Numpy Array
        # x = x.reshape((channels, new_image_width, new_image_height))
        # Normalize
        # x = (x - 128.0) / 128.0
        dataset[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)
    print("All images to array!")

    return dataset

