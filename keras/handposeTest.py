
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import argparse
import imutils
import pickle
from pyimagesearch.preprocessing import aspectawarepreprocessor



model = load_model("handpose.h5")
lb = pickle.loads(open("handposeLabels.pkl","rb").read())
## Grab camera input
cap = cv2.VideoCapture(0)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

# set rt size as 640x480
ret = cap.set(3,640)
ret = cap.set(4,480)

x0 = 400
y0 = 200
height = 240
width = 240
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]

    
    cv2.imshow("handpose",roi)

    roi = cv2.resize(roi,(32,32))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
        

    ############## Keyboard inputs ##################
    key = cv2.waitKey(5) & 0xff
        
    ## Use Esc key to close the program
    if key == 27:
        break
    
    if key == ord("g"):
        preds = model.predict(roi)[0]
        
        idx = np.argmax(preds)
        label = lb.classes_[idx]

        print("predict class:{},probability:{:.4f}".format(label,preds[idx]))
    
    cv2.imshow("Original",frame)
