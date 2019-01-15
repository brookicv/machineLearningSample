
import matplotlib
matplotlib.use("Agg")

import sys
sys.path.append("/home/liqiang/git/machineLearningSample")

from config import dogs_vs_cats_config as config 
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplePreprocessor
from pyimagesearch.preprocessing import patchpreprocessor
from pyimagesearch.preprocessing import meanpreprocessor
from pyimagesearch.callbacks import trainingmonitors
from pyimagesearch.io import hdf5datasetgenerator
from pyimagesearch.nn.conv import alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,
    height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = simplePreprocessor.SimplePreprocessor(227,227)
pp = patchpreprocessor.PatchPreprocessor(227,227)
mp = meanpreprocessor.MeanPreprocessor(means["R"],means["G"],means["B"])
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

trainGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TRAIN_HDF5,128,aug=aug,preprocessors=[pp,mp,iap],classes=config.NUM_CLASSES)
valGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.VAL_HDF5,128,aug=aug,preprocessors=[sp,mp,iap],classes=config.NUM_CLASSES)

print("[INFO] compiling model...")
opt = Adam(lr = 1e-3)
model  = alexnet.AlexNet.build(width=227,height=227,depth=3,classes=config.NUM_CLASSES,reg=0.0002)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH,"{}.png".format(os.getpid())])
callbacks = [trainingmonitors.TrainingMonitor(path)]

model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages //128,
    validation_data = valGen.generator(),validation_steps=valGen.numImages //128,
    epochs = 75,
    max_queue_size=128*2,
    callbacks=callbacks,verbose=1)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH,overwrite=True)

trainGen.close()
valGen.close()