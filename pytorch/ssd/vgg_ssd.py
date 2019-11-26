import torch.nn as nn
import torch
from torchvision.models import vgg16
from torchsummary import summary
from collections import  OrderedDict

import numpy as np
import PIL
from PIL import Image

from l2norm import L2Norm

"""
img = Image.open("1.jpeg")
img = img.resize((300,300))
img = np.array(img,dtype=np.float32)
img = img.transpose((2,0,1))
img = torch.from_numpy(img)
"""

model = vgg16(pretrained=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def conv(in_channels,out_channels,kernel_size,stride=1,padding=1,bn=True):
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding,stride=stride,bias=True),
            nn.ReLU(inplace=True)
        )

class vgg_base_net(nn.Module):

    def __init__(self,num_classes):
        super(vgg_base_net,self).__init__()

        self.num_classes = num_classes
        self.l2norm = L2Norm(512,20)

        self.conv4_3 = nn.Sequential(OrderedDict([
             ("conv1_1",conv(3,64,3)),
            ("conv1_2",conv(64,64,3)),
            ("pool1",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as: 300 -> 150
            ("conv2_1",conv(64,128,3)),
            ("conv2_2",conv(128,128,3)),
            ("pool2",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as: 150 -> 75
            ("conv3_1",conv(128,256,3)),
            ("conv3_2",conv(256,256,3)),
            ("conv3_3",conv(256,256,3)),
            ("pool3",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as  : 75 -> 38
            ("conv4_1",conv(256,512,3)),
            ("conv4_2",conv(512,512,3)),
            ("conv4_3",conv(512,512,3))
        ])) # 38 x 38 x 512

        self.conv5_3 = nn.Sequential(OrderedDict([
            ("pool4",nn.MaxPool2d(2,stride=2,ceil_mode=True)), # downsample, as  : 38 -> 19
            ("conv5_1",conv(512,512,3)),
            ("conv5_2",conv(512,512,3)),
            ("conv5_3",conv(512,512,3))
        ]))

        self.conv6 = nn.Sequential(OrderedDict([
            ("pool5",nn.MaxPool2d(3,stride=1,padding=1)),
            ("conv6",nn.Conv2d(512,1024,kernel_size = 3,padding=6,dilation=6)),
            ("relu6",nn.ReLU(inplace=True))
        ]))

        self.conv7 = nn.Sequential(OrderedDict([
            ("conv7",nn.Conv2d(1024,1024,1)),
            ("relu7",nn.ReLU(inplace=True))
        ])) # 19 x 19 x 1024

        self.conv8 = nn.Sequential(OrderedDict([
            ("conv8_1",conv(1024,256,kernel_size=1,padding=0,bn=False)), # 1 x 1 -> channel reduce
            ("conv8_2",conv(256,512,kernel_size=3,stride=2,padding=1,bn=False)) # 3 x3 with stride = 2 -> downsample
        ])) # 10 x 10 x 512 

        self.conv9 = nn.Sequential(OrderedDict([
            ("conv9_1",conv(512,128,kernel_size=1,padding=0,bn=False)), # 1 x 1 -> channel reduce
            ("conv9_2",conv(128,256,kernel_size=3,stride=2,padding=1))
        ])) # 5 x 5 x 256

        self.conv10 = nn.Sequential(OrderedDict([
            ("conv10_1",conv(256,128,kernel_size=1,padding=0,bn=False)),
            ("conv10_2",conv(128,256,kernel_size=3,stride=1,padding = 0,bn=False))
        ])) # 3 x 3 x 256

        self.conv11 = nn.Sequential(OrderedDict([
            ("conv11_1",conv(256,128,kernel_size=1,padding=0,bn=False)),
            ("conv11_2",conv(128,256,kernel_size=3,stride=1,padding=0,bn=False))
        ])) # 1 x 1 x 256


        # 上面是提取多尺度的特征，共有6个尺度的特征。
        # 下面使用3 x 3 卷积核对特征进行位置回归
        self.loc_1 = nn.Conv2d(512, 4 * 4,kernel_size=3,stride=1,padding=1) # 38 x 38 x 4 x 4 ，每个位置4个default box，每个box需要4个坐标表示
        self.loc_2 = nn.Conv2d(1024,6 * 4,kernel_size=3,stride=1,padding=1) # 19 x 19 x 6 x 4
        self.loc_3 = nn.Conv2d(512, 6 * 4,kernel_size=3,stride=1,padding=1) # 10 x 10 x 6 x 4
        self.loc_4 = nn.Conv2d(256 ,6 * 4,kernel_size=3,stride=1,padding=1) # 5 x 5 x 6 x 4
        self.loc_5 = nn.Conv2d(256, 4 * 4,kernel_size=3,stride=1,padding=1) # 3 x 3 x 4 x 4
        self.loc_6 = nn.Conv2d(256, 4 * 4,kernel_size=3,stride=1,padding=1) # 1 x 1 x 4 x 4

        # 使用3 x 3卷积，进行类别判断
        self.conf_1 = nn.Conv2d(512,4 * self.num_classes,kernel_size=3,stride=1,padding=1) # 38 x 38 x 4 x num_classes,每个default box的类别
        self.conf_2 = nn.Conv2d(1024,6 * self.num_classes,kernel_size=3,stride=1,padding=1) # 19 x 19 x 6 x num_classes
        self.conf_3 = nn.Conv2d(512, 6 * self.num_classes,kernel_size=3,stride=1,padding=1) # 10 x 10 x 6 x num_classes
        self.conf_4 = nn.Conv2d(256,6 * self.num_classes,kernel_size=3,stride=1,padding=1) # 5 x 5 x 6 x num_classes
        self.conf_5 = nn.Conv2d(256,4 * self.num_classes,kernel_size=3,stride=1,padding=1) # 3 x 3 x 4 x num_classes
        self.conf_6 = nn.Conv2d(256,4 * self.num_classes,kernel_size=3,stride=1,padding=1) # 1 x 1 x 4 x num_classes

    def forward(self, x):
        x = self.conv4_3(x)
        feature_map_1 = self.l2norm(x) # 38 x 38 x 512

        x = self.conv5_3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        feature_map_2 = x # 19 x 19 x 1024

        x = self.conv8(x)
        feature_map_3 = x # 10 x 10 x 512
        
        x = self.conv9(x)
        feature_map_4 = x # 5 x 5 x 256

        x = self.conv10(x)
        feature_map_5 = x # 3 x3 x 256

        x = self.conv11(x)
        feature_map_6 = x # 1 x 1 x 256

        # 边框回归，和类别判断
        loc1 = self.loc_1(feature_map_1).permute((0,2,3,1)).contiguous() # 1 x 16 x 38 x 38 -> 1 x 38 x 38 x 16
        conf1 = self.conf_1(feature_map_1).permute((0,2,3,1)).contiguous() # 38 x 38 x 4 x num_classes

        loc2 = self.loc_2(feature_map_2).permute((0,2,3,1)).contiguous()
        conf2 = self.conf_2(feature_map_2).permute((0,2,3,1)).contiguous()

        loc3 = self.loc_3(feature_map_3).permute((0,2,3,1)).contiguous()
        conf3 = self.conf_3(feature_map_3).permute((0,2,3,1)).contiguous()

        loc4 = self.loc_4(feature_map_4).permute((0,2,3,1)).contiguous()
        conf4 = self.conf_4(feature_map_4).permute((0,2,3,1)).contiguous()

        loc5 = self.loc_5(feature_map_5).permute((0,2,3,1)).contiguous()
        conf5 = self.conf_5(feature_map_5).permute((0,2,3,1)).contiguous()

        loc6 = self.loc_6(feature_map_6).permute((0,2,3,1)).contiguous()
        conf6 = self.conf_6(feature_map_6).permute((0,2,3,1)).contiguous()

        return [loc1, loc2, loc3, loc4, loc5,loc6],[conf1, conf2, conf3, conf4, conf5,conf6]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base = vgg_base_net(2)

base.to(DEVICE)

x = torch.randn((1,3,300,300),dtype=torch.float32)
x = x.to(DEVICE)

base.eval()
loc,conf = base(x)

print(loc[0].shape)
print(conf[0].shape)
    
