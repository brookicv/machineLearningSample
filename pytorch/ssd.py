import torch.nn as nn
import torch
from torchvision.models import vgg16
from torchsummary import summary
from collections import  OrderedDict

import numpy as np
import PIL
from PIL import Image

img = Image.open("1.jpeg")
img = img.resize((300,300))
img = np.array(img,dtype=np.float32)
img = img.transpose((2,0,1))
img = torch.from_numpy(img)

model = vgg16(pretrained=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def conv(in_channels,out_channels,kernel_size,strides=1,padding=1,bn=True):
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,bias=True),
            nn.ReLU(inplace=True)
        )

class vgg_base_net(nn.Module):

    def __init__(self):
        super(vgg_base_net,self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ("conv1",conv(3,64,3)),
            ("conv2",conv(64,64,3)),
            ("pool1",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as: 300 -> 150
            ("conv3",conv(64,128,3)),
            ("conv4",conv(128,128,3)),
            ("pool2",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as: 150 -> 75
            ("conv5",conv(128,256,3)),
            ("conv6",conv(256,256,3)),
            ("conv7",conv(256,256,3)),
            ("pool3",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as  : 75 -> 38
            ("conv8",conv(256,512,3)),
            ("conv9",conv(512,512,3)),
            ("conv10",conv(512,512,3)),
            ("pool4",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as  : 38 -> 19
            ("conv11",conv(512,512,3)),
            ("conv12",conv(512,512,3)),
            ("conv13",conv(512,512,3)),
            ("pool5",nn.MaxPool2d(3,stride=1,padding=1)),
            ("conv14",nn.Conv2d(512,1024,kernel_size = 3,padding=6,dilation=6)),
            ("relu14",nn.ReLU(inplace=True)),
            ("conv15",nn.Conv2d(1024,1024,1)),
            ("relu15",nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        x = self.model(x)

        return x

base = vgg_base_net()
base.to(DEVICE)

summary(base,(3,300,300))
    
