import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import BatchNormalization,AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import Conv2D,Activation,Dropout,Flatten
from tensorflow.keras.layers import Input,Dense,concatenate

def shallow_sequential(width,height,depth,classes):
    # initialize the model along the input shape
    # "chanels last" ordering
    model = Sequential()
    inputShape = (height,width,depth)

    # first layer
    # 卷积网络的第一层要提供，input shape， 后续卷基层则无需提供输入的shape，由上一层计算得到
    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

def mini_goolenet_functional(width,height,depth,classes):
    
    def conv_module(x,K,Kx,Ky,stride,chanDim,padding="same"):
        x = Conv2D(K,(kX,kY),strides=stride,padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        return x

    def inception_module(x,numK1x1,numK3x3,chanDim):

        conv_1x1 = conv_module(x,numK1x1,1,1,(1,1),chanDim)
        conv_3x3 = conv_module(x,numK3x3,3,3,(3,3),chanDim)
        x = concatenate([conv_1x1,conv_3x3],axis=chanDim)

        return x

    def down_sample_module(x,K,chanDim):
        
        conv_3x3 = conv_module(x,K,3,3,(2,2),chanDim,padding="valid")
        pool = MaxPooling2D((3,3),(2,2))(x)
        x = concatenate([conv_3x3,pool],axis=chanDim)

        return x

    
    inputShape = (height,width,depth)
    chanDim = -1

    inputs = Input(shape=inputShape)
    x = conv_module(inputs,96,3,3,(1,1),chanDim)

    # two inception modules
    x = inception_module(x,32,32,chanDim)
    x = inception_module(x,32,48,chanDim)
    x = down_sample_module(x,80,chanDim)

    # four Inception modules
    x = inception_module(x,112,48,chanDim)
    x = inception_module(x,96,64,chanDim)
    x = inception_module(x,80,80,chanDim)
    x = inception_module(x,48,96,chanDim)
    x = down_sample_module(x,96,chanDim)

    x = inception_module(x,176,160,chanDim)
    x = inception_module(x,176,160,chanDim)
    x = AveragePooling2D((7,7))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)

    # intput,output
    model = Model(inputs,x ,name="minigoolenet")

    return model