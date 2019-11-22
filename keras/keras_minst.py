
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets

import matplotlib.pyplot as plt 
import numpy as np 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading MINST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")

data = dataset.data.astype("float") / 225.0
(trainX,testX,trainY,testY) = train_test_split(data,dataset.target,test_size=0.25)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# NN architecture 784-256-128-10 using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

#train the model using SGD
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
H = model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=128)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100),H.history["acc"],label="train_acc")
plt.plot(np.arange(0,100),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


