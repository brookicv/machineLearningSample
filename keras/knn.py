
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

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% 
# of the data for training and the remaining 25% for testing
(trainX,testX,trainY,testY) = train_test_split(data,labels,
    test_size = 0.25,random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluuating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))