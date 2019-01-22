from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K 

class ResNet:
    @staticmethod
    def residual_module(data,K,stride,chanDim,red=False,
        reg=0.0001,bnEps=2e-5,bnMom=0.9):
        '''
        The specific implementation of ResNet was inspired by He et al. in their Caffe distribution

        Args:
            data: The input to the residual module
            K: The number of filters that will be leraned by the final CONV int the bottleneck,
               the first two CONV layers will learn K / 4 filters
            stride: The strides of convolution,to reduce spatial dimensions of volumen without resorting to max pooling
            chanDim: Define the axis that perform batch normalization
            red: Control whether reducing spatial dimentions(True) or not(False)
            reg: Regularization
            bnEps: Avoiding "division by zero",in Keras ,it defaults to 0.001
            bnMom: The momentum for the moving average,it defaults to 0.99 in Keras
        '''

        # the shortcut branch of the ResNet module should be the initialize as the input(identity) data
        shortcut = data 

        # the first block of the ResNet module are the 1x1 CONVS
        bn1 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)

        # the second block ot the ResNet module are the 1x1 CONVS
        bn2 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2= Activation("relu")(bn2)
        covn2 = Conv2D(int(K * 0.25),(3,3),strides=stride,padding="same",use_bias=False,kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVS
        bn3 =  bn2 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(covn2)
        act3= Activation("relu")(bn3)
        conv3 = Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size,apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K,(1,1),strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3,shortcut])

        return x
    
    @staticmethod
    def build(width,height,depth,classes,stages,filters,reg=0.0001,bnEps=2e-5,bnMom=0.9,dataset="cifar"):

        inputShape=(height,width,depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(inputs)

        if dataset == "cifar":
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg))(x)
        
        for i in range(0,len(stages)):
            
            stride=(1,1) if i == 0 else (2,2)
            x = ResNet.residual_module(x,filters[i + 1],stride,chanDim,red=True,bnEps=bnEps,bnMom=bnMom)

            for j in range(0,stages[i] - 1):
                x = ResNet.residual_module(x,filters[i + 1],(1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
            
        
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8,8))(x)

        x = Flatten()(x)
        x = Dense(classes,kernel_regularizer=l2(reg))(x)
        x =  Activation("softmax")(x)

        model = Model(inputs,x,name="resnet")

        return model

