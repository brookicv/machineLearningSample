from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.datasets import simpleDatasetLoader
from pyimagesearch.nn.conv import fcheadnet
from keras.optimizers import RMSprop
from keras.optimizers import SGD 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model 
from imutils import paths
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-m","--model",required=True,help="path to output model")
ap.add_argument("-l","--lablebin",required=True,help="path to output label binarizer")

args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,
    shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

sp = simplePreprocessor.SimplePreprocessor(224,224)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl = simpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.astype("float")/255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))
headModel = fcheadnet.FCHeadNet.build(baseModel,len(classNames),256)

# place the head FC model on top of the base model
# this model will become the actual model we will train
model = Model(inputs=baseModel.input,outputs=headModel)

# loop over all layers in the base model and freeze them 
# so they will not be updated during the training process
for layer in baseModel.layers[:15]:
    layer.trainable = False

epochs = 1

print("[INFO] compiling model...")
opt = RMSprop(lr = 0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=32),validation_data=(testX,testY),
    epochs=epochs,steps_per_epoch=len(trainX)//32,verbose=1)

# save the model of network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save hte label binarizer 
print("[INFO] serializing label binarizer...")
f = open(args["lablebin"],"wb")
pickle.dump(lb,f)
f.close()

# evaluate the network
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
    target_names=lb.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,epochs),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,epochs),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,epochs),H.history["acc"],label="acc")
plt.plot(np.arange(0,epochs),H.history["val_acc"],label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

