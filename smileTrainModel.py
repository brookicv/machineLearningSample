# -*- coding:utf-8 -*-

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv import lenet
from imutils import paths
import matplotlib.pyplot as pyplot
import numpy as  np 
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset of faces")
ap.add_argument("-m","--model",required=True,help="path to output model")
args = vars(ap.parse_args())

data = []
labels=[]

for imagePath in sorted(list(paths.list_image(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image,width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data,dtype="float")/255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels),2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

