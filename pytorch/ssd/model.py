from torch import nn as nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGGBase(nn.Module):

    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # downsample 2
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.covn2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample 2
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # downsample 2,大小向上取整
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample 2
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 大小不变，padding=1，stride=1
        
        # 替换原结构的FC6,FC7
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # 空洞卷积，扩大感受野
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # 1 x 1

        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)  # (N,64,150,150)
        
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.poo2(out)  # (N,128,75,75)
        
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)  # (N,256,38,38)
        
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feats = out  # (N,512,38,38)
        out = self.pool4(out)  # (N,512,19,19)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)  # (N,512,19,19) ,没有减小尺度
        
        out = F.relu(self.conv6(out))  # (N,1024,19,19) 
        
        conv7_feats = F.relu(self.conv7(out))  #(N,1024,19,19)
        
        return conv4_3_feats,conv7_feats
        
        
    def load_pretrained_layers(self):
        """
        加载VGG的预训练模型
        对FC6 - conv6，FC7 - conv7的参数，进行变换
        """

        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict)

        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]] # 按照顺序加载vgg

        # 转换fc6,fc7的参数,将采样的方式 以适合conv6,conv7
        #fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])
        
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(conv_fc7_weight,m = [4,4,None,None])
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])
        
        self.load_state_dict(state_dict)

        print("\n Loaded base model. \n")


class AuxiliaryConvolution(nn.Module):
    """
    在vgg conv7的基础上，继续提取更高层次的特征
    """

    def __init__(self):
        super(AuxiliaryConvolution, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # 减少通道 1 x 1，各通道信息进行融合
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 减少尺度，stride = 2 , 10 x 10
        
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 5 x 5
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  #  3 x 3
        
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # 1 x 1
        
        # 初始化参数
        self.init_conv2d()

    def init_conv2d(self):

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias)


    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv8_2(out))
        conv9_2_feats = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out

        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        conv11_2_feats = out

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
        
class PredictionConvolution(nn.Module):
    """
    进行边框回归和分类的卷积

    预测边框相对与预设的prior box的偏移量； 预测每个边框的类别
    """
    def __init__(self, n_classes):
        super(PredictionConvolution, self).__init__()
        
        self.n_claases = n_classes

        n_boxes = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2':4
        }

        # 边框预测
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # 分类
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)


        # 初始化参数
        self.init_conv2d()

    def init_conv2d(self):

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        512 x 38 x 38, 1024 x 19 x 19 ,512 x 10 x 10,256 x 5 x 5 ,256 x 3 x 3 , 256 x 1 x 1
        """

        batch_size = conv4_3_feats.size(0)

        # 预测边框相对于prior box的偏移量
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats).permute((0, 2, 3, 1)).contiguous()  # (N,16,38,38) -> (N,38,38,16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # N,38 x 38 ,4
        
        l_conv7 = self.loc_conv7(conv7_feats).permute((0, 2, 3, 1)).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N,19 x 19 ,4)
        
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats).permute((0, 2, 3, 1)).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats).permute((0, 2, 3, 1)).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats).permute((0, 2, 3, 1)).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats).permute((0, 2, 3, 1)).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)

        # 分类
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_claases)
        
        c_conv7 = self.cl_conv7(conv7_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_claases)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, 4)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats).permute((0, 2, 3, 1)).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, 4)

        # Total
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # N ,8732,4
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)  # N,8732,n_classes
        
        return locs, classes_scores
        
class SSD300(nn.Module):

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolution()
        self.pred_convs = PredictionConvolution(n_classes)

        self.prior_cxcy = self.create_prior_boxes()

        # conv4_3提取的是低层的特征，而且尺度较大，使用l2 norm对其进行归一化，并rescale
        # rescale 可以在训练的过程中进行学习
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)) # 512 通道
        nn.init.constant_(self.rescale_factors,20)

    def forward(self, image):
        
        conv4_3_feats, conv7_feats = self.base(image)
        
        # rescale conv4_3 after l2norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        # (n,512,10,10),(n,256,5,5),(n,256,3,3),(n,256,1,1)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
        
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
        
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes