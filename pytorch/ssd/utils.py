
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

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision

def cxcy_to_xy(cxcy):
    """
    将边框的表示形式由(c_x,c_y,w,h)的形式转换为(x_min,y_min,x_max,y_max)
    """
    return torch.cat([cxcy[:,:2] - (cxcy[:,2:] / 2),
                    cxcy[:,:2] + (cxcy[:,2:] / 2)],dim=1)

def cxcy_to_gcxgcy(cxcy,prior_box):
    """
    cxcy表示的边框相对于prior box的偏移量
    """
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    # 10 和 5 是经验值，用来缩放位置回归的梯度
    return torch.cat([(cxcy[:,:2] - prior_box[:,:2]) / (prior_box[:,2:] / 10),
                    torch.log(cxcy[:,2:] / prior_box[:,2:]) * 5],dim=1)

def gcxcy_to_cxcy(gcxcy,prior_box):
    """
    根据相对于prior box的偏移量，计算边框
    """
    return torch.cat([gcxcy[:,:2] * prior_box[:,2:] / 10 + prior_box[:,:2],
                    torch.exp(gcxcy[:,2:] / 5) * prior_box[:,2:]],dim=1)

# 边框的表示形式为 x_min,y_min,x_max,y_max
def find_intersection(set_1,set_2):

    lower_bounds = torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:,2:].unsqueeze(1),set_2[:,2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds,min=0)

    return intersection_dims[:,:,0] * intersection_dims[:,:,1]

def find_jaccard_overlap(set_1,set_2):
    
    intersection = find_intersection(set_1,set_2)

    area_set_1 = (set_1[:,2] - set_1[:,0]) * (set_1[:,3] - set_1[:,1])
    area_set_2 = (set_2[:,2] - set_2[:,0]) * (set_2[:,3] - set_2[:,1])

    union = area_set_1.unsqueeze(1) + area_set_2.unsqueeze(0) - intersection

    return intersection / union


def adjust_learning_rate(optimizer,scale):

    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * scale
    
    print(print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],)))

def accuracy(scores,targets,k):
    """
    计算 top-k精度
    scores: 预测的score
    targets: true labels
    """
    batch_size = targets.size(0)
    _,ind = scores.topk(k,1,True,True)
    correct = ind.eq(targets.view(-1,1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100 / batch_size)

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



