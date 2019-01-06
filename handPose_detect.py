from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image that to recognize")
ap.add_argument("-m","--model",required=True,help="path to pre-trained smile detector CNN")

args = vars(ap.parse_args())

model = load_model(args["model"])

im = cv2.imread(args["image"])
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = cv2.resize(im,(32,32))
im = im.astype("float") / 255.0
im = img_to_array(im)
im = np.expand_dims(im,axis=0)

(ok,peace,punch,stop) = model.predict(im)[0]

print("ok:{:.4f},peace:{:.4f},punch:{:.4f},stop:{:.4f}".format(ok,peace,punch,stop))