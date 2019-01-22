
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K 

class MiniGoogLeNet:

    @staticmethod
    def conv_module(x,K,kX,kY,stride,chanDim,padding="same"):
        '''
        Conv module si responsible for applying a convolution,followed by a batch normalization,
        and then finally an activation

        Args:
            x: The input layer to the function
            K: The number of filter CONV layer is going to leran
            kX,kY: The size of each of the K filters that will be leraned
            stride: The stride of the CONV layer
            chanDim: The channel dimension,"channels first" or "channels last"
            padding:The type of padding to be applied to the CONV layer
        '''
        # define a CONV => BN => RELU pattern
        x = Conv2D(K,(kX,kY),strides=stride,padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        return x

    @staticmethod
    def inception_module(x,numK1x1,numK3x3,chanDim):
        '''
        Inception module perform two sets of convolutions - a 1x1 CONV and a 3x3 CONV.
        The two convolutions will be performed in parallel and the resulting features
        concatenated across the channel dimension

        Args:
            x: The input layer to the function
            numK1x1: The number of 1x1 filter
            numK3x3: The number of 3x3 filter
            chanDim: The channel dimension
        '''

        conv_1x1 = MiniGoogLeNet.conv_module(x,numK1x1,1,1,(1,1),chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x,numK3x3,3,3,(1,1),chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis=chanDim)

        return x

    @staticmethod
    def downsample_module(x,K,chanDim):
        '''
        Downsample module is reponsible for reducing the saptial dimensions of an input volume

        Args:
            x: The input layer to the funcion
            K: The numbler of filter will be learn
            chanDim: The channel dimension
        '''
        conv_3x3 = MiniGoogLeNet.conv_module(x,K,3,3,(2,2),chanDim,padding="valid")
        pool = MaxPooling2D((3,3),strides=(2,2))(x)
        x = concatenate([conv_3x3,pool],axis=chanDim)

        return x

    @staticmethod
    def build(width,height,depth,classes):
        '''
        Put all modules together

        Args:
            width,height,depth: Input shape
            classes: The number of classes
        '''
        inputShape = (height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1
        
        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs,96,3,3,(1,1),chanDim)

        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x,32,32,chanDim)
        x = MiniGoogLeNet.inception_module(x,32,48,chanDim)
        x = MiniGoogLeNet.downsample_module(x,80,chanDim)

        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x,112,48,chanDim)
        x = MiniGoogLeNet.inception_module(x,96,64,chanDim)
        x = MiniGoogLeNet.inception_module(x,80,80,chanDim)
        x = MiniGoogLeNet.inception_module(x,48,96,chanDim)
        x = MiniGoogLeNet.downsample_module(x,96,chanDim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = MiniGoogLeNet.inception_module(x,176,160,chanDim)
        x = AveragePooling2D((7,7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs,x,name="googlenet")

        return model


    