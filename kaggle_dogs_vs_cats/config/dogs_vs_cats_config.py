
#define the paths to the image directory
IMAGES_PATH = "/home/liqiang/openSources/Deep-Learning-For-Computer-Vision/datasets/animals"

NUM_CLASSES = 3
NUM_VAL_IMAGES = 600
NUM_TEST_IMAGES = 600

# define the path to the output training,validation,and testing HDF5 files
TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VAL_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

# path to output model file
MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

# define the path to the dataset mean 
DATASET_MEAN = "output/dogs_vs_cats_means.json"

# define the path to the output directory used for storing plots,classification,etc.
OUTPUT_PATH = "output"