
# -*- coding:utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.datasets import simpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Path to input dataset")
ap.add_argument("-k","--neighbors",type=int,default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j","--jobs",type=int,default=1,help="# of jobs for k-NN distance(-1 uses all available cores)")

args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplePreprocessor.SimplePreprocessor(32,32)
sdl = simpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.reshape((data.reshape[0]),3072)

print("[INFO] features matrix:{:.1f}MB".format(data.nbytes / (1024 * 1000.0)))
