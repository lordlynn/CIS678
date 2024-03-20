# This code reads the data files and creates a test/training split
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageOps 
import random

def flattenImages(images):
    for image in range(len(images)):
        # Flatten numpuy arrays
        images[image] = images[image].flatten()

    return images

def getDataset(flatten=False):
    # Read images from train folder excluding truth.txt
    directory = "./dataset/train/"
    fileNames = [f for f in listdir(directory) if isfile(join(directory, f))]
    fileNames.remove("truth.txt")                               

    trainingImages = [np.asarray(ImageOps.grayscale(Image.open(directory + name))) for name in fileNames]
    if (flatten):
        trainingImages = flattenImages(trainingImages)

    # Read truth.txt from train folder
    with open(directory + "truth.txt", "r") as fp:
        trainingTruth = fp.readlines()
    
    trainingTruth = [[int(line.strip("\n"))] for line in trainingTruth]

    ############################################################################################

    # Read images from test folder excluding truth.txt
    directory = "./dataset/test/"
    fileNames = [f for f in listdir(directory) if isfile(join(directory, f))]
    fileNames.remove("truth.txt")
    
    testImages = [np.asarray(ImageOps.grayscale(Image.open(directory + name))) for name in fileNames]
    if (flatten):
        testImages = flattenImages(testImages)

    # Read truth.txt from test folder
    with open(directory + "truth.txt", "r") as fp:
        testTruth = fp.readlines()
    
    testTruth = [[int(line.strip("\n"))] for line in testTruth]


    return trainingImages, trainingTruth, testImages, testTruth


def readFiles():
    directory = "./dataset/no/"

    # From: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    # Read files in no folder (normal images of brain)
    fileNames = [f for f in listdir(directory) if isfile(join(directory, f))]

    # Read files and convert them tp numpy arrays ***Format is [y][x][R/G/B]
    normal = [np.asarray(ImageOps.grayscale(Image.open(directory + name))) for name in fileNames]

    directory = "./dataset/yes/"

    # Read files in yes folder (images of brain with tumor)
    fileNames = [f for f in listdir(directory) if isfile(join(directory, f))]

    abnormal = [np.asarray(ImageOps.grayscale(Image.open(directory + name))) for name in fileNames]

    
    return normal, abnormal


# This could definitely be done in fewer steps but it works
def resizeImages(images, shape):
    flatShape = shape[0] * shape[1]

    for image in range(len(images)):
        # Convert back to PIL image
        images[image] = Image.fromarray(images[image])
        
        # Downsample all images to the smallest shape, use antialiasing filter
        images[image] = images[image].resize(shape, resample=Image.LANCZOS)
        
    return images


# If no seed is 0 and split percentage is same, this will alwasy create same split
# To create new splits provide a seed to randomize the mixing process
def trainTestSplit(split, Rseed=0):
    split = split / 100
    random.seed(Rseed)
    
    normal, abnormal = readFiles()

    shapes = [image.shape for image in normal] + [image.shape for image in abnormal] 
    shape = min(shapes)

    normal = resizeImages(normal, shape)
    abnormal = resizeImages(abnormal, shape)

    imagesForTraining = int((len(normal) + len(abnormal)) * split)
    normalMax = len(normal)
    abnormalMax = len(abnormal)


    trainingImages = []
    trainingTruth = []
    for i in range(imagesForTraining):                              # Iterate as many times as images we need for training
        choice = random.random()

        # This ensures that the same number of normal/abnormal 
        #    images are in trainign and test datasets
        if (len(normal) <= normalMax * (1-split)):
            choice = 1
        elif (len(abnormal) <= abnormalMax * (1-split)):
            choice = 0


        # Add a normal image to the traning data
        if (choice < 0.5):
            index = int(random.random() * len(normal))              # Calculate a random index to get image from
            trainingImages.append(normal[index])                    # Append image to training data list
            trainingTruth.append(0)                                 # Append value to truth list based on file 
            del normal[index]                                       # Delete image from original list
        
        # Add abnormal image to the training data
        else:
            index = int(random.random() * len(abnormal))       
            trainingImages.append(abnormal[index])            
            trainingTruth.append(1)
            del abnormal[index]                               


    # since training data was deleted from the orginal list,
    #   test data is simply the original list combined    
    
    testImages = normal + abnormal

    # Create test truth list. 0 - NORMAL, 1- ABNORMAL
    testTruth = [0] * len(normal) + [1] * len(abnormal)
    
    # ***** MAKE SURE YOU CLEAR THE TRAIN AND TEST DIRECTORIES BEFORE WRITING

    # Save training data to folder train
    directory = "./dataset/train/"
    for image in range(len(trainingImages)):
        trainingImages[image].save(directory + str(image) + ".jpg")

    with open(directory + "truth.txt", "w") as fp:
        for d in trainingTruth:
            fp.write(str(d) + "\n")


    # Save test data to folder train
    directory = "./dataset/test/"
    for image in range(len(testImages)):
        testImages[image].save(directory + str(image) + ".jpg")
    
    with open(directory + "truth.txt", "w") as fp:
        for d in testTruth:
            fp.write(str(d) + "\n")

    return trainingImages, trainingTruth, testImages, testTruth



# trainTestSplit(75)