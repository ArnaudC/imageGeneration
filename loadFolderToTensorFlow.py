import os, sys

import random
import numpy as np
from cv2 import cv2 # Need to install an older version on windows : 'pip install opencv-python=3.3.0.9'

# Expected output (self.img_rows, self.img_cols, self.channels)
# channels = 3 colors : R, G, B.
def loadFolderToTensorFlow(folder, image_width, image_height, ratio, resizedFolder, channels = 3, percentageOfImagesToKeep = 100, minFiles = 3):

    # Load all files
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    filesNb = len(onlyfiles)
    if (filesNb == 0):
        print ("No file found in {folder}".format(folder=folder))
        exit()
    if (percentageOfImagesToKeep < 100):
        random.shuffle(onlyfiles)
        lastIndex = int((filesNb - 1) * (percentageOfImagesToKeep / 100))
        lastIndex = minFiles if (lastIndex == 0) else lastIndex
        onlyfiles = onlyfiles[0:lastIndex]
        print("Percentage of images to keep : {p} %. Using {ni} of {t} images.".format(p = percentageOfImagesToKeep, t=filesNb,  ni = lastIndex + 1))
    else:
        print("Working with {0} images".format(filesNb))
    train_files = []
    y_train = []
    for _file in onlyfiles:
        train_files.append(_file)
        start = _file.rfind('_') + 1
        end = len(_file) - 4
        imgNb = _file[start:end]
        y_train.append(int(imgNb))
    nFiles = len(train_files)
    print("Files in train_files: %d" % nFiles)

    # Resize image : (height, width) / ratio
    # image_width = 773 # Original Dimensions
    # image_height = 1080
    # ratio = 4 # Resize pictures
    new_image_height = int(image_height / ratio)
    new_image_width = int(image_width / ratio)
    # channels = 3
    # nb_classes = 1

    # Create dataset
    dataset = np.ndarray(shape=(nFiles, new_image_height, new_image_width, channels), dtype=np.float32)

    # Clean resized folder
    removeAllFilesInFolder(resizedFolder)

    i = 0
    for _file in train_files:
        img = cv2.imread(folder + "/" + _file)

        # Display the image
        # img.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img_new = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        x = cv2.resize(img, dsize=(new_image_width, new_image_height), interpolation=cv2.INTER_CUBIC)
        if (i <= (minFiles - 1)):
            cv2.imwrite(resizedFolder + _file, x) # See resized images

        # img = load_img(folder + "/" + _file)  # this is a PIL image
        # img.thumbnail((new_image_width, new_image_height))
        # x = img_to_array(img) # Convert to Numpy Array
        # x = x.reshape((channels, new_image_width, new_image_height))
        dataset[i] = x
        i += 1
        
        if i % 250 == 0:
            p = round(100 * i / nFiles, 2)
            print("{nImg} images to array, {percent} % done".format(nImg = i, percent = p))
    print("All images to array!")

    return (dataset, new_image_height, new_image_width)


# https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
def removeAllFilesInFolder(folder):
    for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
