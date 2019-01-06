# -*- coding:utf-8 -*-

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv import lenet
from imutils import paths
import matplotlib.pyplot as plt
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

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
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
# one-hot encoding
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels),2)

# hand the class imbalance
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=42)

# initialize the model
print("[INFO] compiling model ....")
model = lenet.LeNet.build(width=28,height=28,depth=1,classes=2)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

# train
print("[INFO] training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),class_weight=classWeight,batch_size=64,epochs=15,verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=le.classes_))

# save model
print("[INFO] saving model...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,15),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,15),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,15),H.history["acc"],label="acc")
plt.plot(np.arange(0,15),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

