from kaggle_dogs_vs_cats.config import dogs_vs_cats_config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import aspectawarepreprocessor
from pyimagesearch.io import hdf5datasetwriter
from imutils import paths
import numpy as np 
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(dogs_vs_cats_config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[2].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
