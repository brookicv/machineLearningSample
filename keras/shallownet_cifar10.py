from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from pyimagesearch.nn.conv import shallownet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

print("[INFO] loading CIFAR-10 data...")
((trainX,trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX  = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# labels name
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

# initialize the network
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = shallownet.ShallowNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

# train
print("[INFO] training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=32,epochs=100,verbose=1)

# evaluate the network
print("[INFO] evaluating network ....")
predicitions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predicitions.argmax(axis=1),target_names=labelNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,40),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,40),H.history["acc"],label="acc")
plt.plot(np.arange(0,40),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()