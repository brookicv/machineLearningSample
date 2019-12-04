
import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def parse_annotation(annotation_path):
    """
        解析VOC格式的标记文件
    """
    tree = Et.parse(annotation_path)
    root = tree..getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")

        label = object.find("name").text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}
    
def create_data_list(voc07_path, voc12_path, output_folder):
    
    voc07_path = os.path.abspath()
    voc12_path = os.path.abspath()

    train_images = list()
    train_objects = list()
    n_objects = 0

    # training data 
    for path in [voc07_path, voc12_path]:
        # 找到每张图像的id
        with open(os.path.join(path, "ImageSets/Main/trainva.txt")) as f:
            ids = f.read().splitlines()

        for id in ids:
            # 解析每张图片的标记文件
            objects = parse_annotation(os.path.join(path, "Annotations", id + ".xml"))
            if len(objects) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, "JPEGImage", id + ".jpg"))
            
    assert len(train_objects) == len(train_images)

    # 结果保存到文件中
    with open(os.path.join(output_folder, "TRAIN_images.json"), "w") as j:
        json.dump(train_images, j)
        
    with open(os.path.join(output_folder,"TRAIN_object.json"),"w") as j:
        json.dump(train_objects, j)
        
    with open(os.path.join(output_folder, "label_map.json"), "w") as j:
        json.dump(label_map, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    
     # 测试集
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Find IDs of images in validation data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

def decimate(tensor, m):
    
    assert tensor.dim() == len(m)
    for d in range(len(tensor.dim())):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.range(start=0, end=tensor.size(d), step=m[d]).long())
            
    return tensor