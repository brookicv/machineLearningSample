from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print("[INFO] loading MNIST (sample) dataset...")

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
digits = datasets.load_digits()

data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())

print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")

nn = NeuralNetwork([data.shape[1], 32,16, 10])

print ("[INFO] {}".format(nn))

nn.fit(trainX, trainY, epochs=1000)

print ("[INFO] evaluating network...")

# Predict the test set.
# Note that the predictions array has a shape of (450, 10) because there are
# 450 samples in the test set, and 10 possible outcomes (each value will
# represent the probability)
predictions = nn.predict(testX)

# Note: the argmax function will return the index of the label with the highest
# predicted probability
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))