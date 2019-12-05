import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from augmentations import transform

class PascalVocDataset(Dataset):
    """
    PyTorch的dataset，实现数据的加载
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # read objects(边框，类别，difficulties标记)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.FloatTensor(objects["labels"])
        difficulties = torch.FloatTensor(objects["difficulties"])

        # 不使用困难样本训练
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        
        # apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        由于每张图像包含的目标个数不一样，在将数据传给dataloader前，需要一个校对函数
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        
        return images,boxes,labels,difficulties # tensor (N,3,300,300)