
# -*- coding:utf-8 -*-

import numpy as np 
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None :
            self.preprocessors = []

    def load(self,imagePaths,depth=3,verbose=-1):
        data = []
        labels = []

        for (i,imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)

            if depth != 3:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # Path to dataset/{class}/{image}.jpg
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                # loop,apply each to the image
                for p in self.preprocessors:
                    image = p.process(image)

            data.append(image)
            labels.append(label)

            # show an 'verbose' message
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))

        # return a tuple of data and labels
        return (np.array(data),np.array(labels))