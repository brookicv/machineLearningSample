from itertools import  product
import torch 

import math
from math import  sqrt
import numpy as np
 
def generate_prior_box():

    fmap_dims = {"conv4_3": 38,
                "conv7": 19,
                "conv8_2": 10,
                "conv9_2": 5,
                "conv10_2": 3,
                "conv11_2": 1}  # feature maps size
    obj_scales = {"conv4_3": 0.1,
                "conv7": 0.2,
                "conv8_2": 0.375,
                "conv9_2": 0.55,
                "conv10_2": 0.725,
                "conv11_2": 0.9}  # scale
    aspect_ratios = {"conv4_3": [1.,2.,0.5],
                    "conv7": [1.,2.,3.,0.5,0.333],
                    "conv8_2": [1.,2.,3.,0.5,0.333],
                    "conv9_2": [1.,2.,3.,0.5,0.333],
                    "conv10_2": [1.,2.,0.5],
                    "conv11_2": [1.,2.,0.5]}  # ratios

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):

                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap] # 中心点

                for ratio in aspect_ratios[fmap]:

                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                    
                    # ratio = 1.0的情况下，添加一个额外的prior box
                    if ratio == 1.0:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        except IndexError:
                            additional_scale = 1.0  # 最后一个scale 
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])
                        
    prior_boxes = torch.FloatTensor(prior_boxes)  # (8732,4)
    prior_boxes.clamp_(0,1) # 裁剪

    # prior_boxes = np.clip(prior_boxes,0,1)

    return prior_boxes



    