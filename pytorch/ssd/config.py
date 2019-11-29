import os.path

sk = [15,30,60,111,162,231,264]
feature_map = [28,19,10,5,3,1]
steps = [8,16,32,64,100,300]
image_size = 300
aspect_ratios = [[2],[2,3],[2,3],[2,3],[2],[2]]
MEANS = (104,117,123) 
batch_size = 32
data_load_number_workder = 0
lr = 1e-3
momentum = 0.9
weight_decacy = 5e-4
gamma = 0.1
VOC_ROOT  = ""
dataset_root = VOC_ROOT
USE_CUDA = True
lr_steps = (80000)
max_iter = 120000
class_num = 21


s_min = 0.2
s_max = 0.9

scale_list = []
aspect_ratios_list = [1,2,3,1/2,1/3]

for i in range(1,7):
    s = s_min + (s_max - s_min)/(6 - 1) * (i - 1)
    for ar in aspect_ratios_list:
        w = s * sqrt(ar)
        h = s / sqrt(ar)