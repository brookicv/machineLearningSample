from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.datasets import simpleDatasetLoader
from pyimagesearch.nn.conv import lenet
from pyimagesearch.nn.conv import minivggnet
from imutils import paths
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to output model")

args = vars(ap.parse_args())

inputWidth = 32
inputHeight = 32
inputDepth = 3

trainingEpochs = 100

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplePreprocessor.SimplePreprocessor(inputWidth,inputHeight)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = simpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(imagePaths,depth = inputDepth,verbose=100)
data = data.astype("float") / 255.0

classNames = sorted(list(set(labels)))
classes = len(classNames)
print(classNames)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# initialize the model
print("[INFO] compiling model ....")
#model = lenet.LeNet.build(width=32,height=32,depth=1,classes=4)
model = minivggnet.MiniVGGNet.build(width=inputWidth,height=inputHeight,depth=inputDepth,classes=classes)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# train
print("[INFO] training network...")
H = model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=trainingEpochs,verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=classNames))

# save model
print("[INFO] saving model...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,trainingEpochs),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,trainingEpochs),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,trainingEpochs),H.history["acc"],label="acc")
plt.plot(np.arange(0,trainingEpochs),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()