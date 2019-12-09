import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from dataset import PascalVocDataset
from utils import *

data_folder = './'
keep_diffcult = True

n_classes = len(label_map)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = None
batch_size = 8
start_epoch = 0
epochs = 200
epochs_since_improvement = 0
best_loss = 0
workder = 4
print_freq = 200
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
grad_clip = None

cudnn.benchmark = True

def main():
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint

    if checkpoint is None:
        model = SSD300(n_classes=n_classes)

        biases = list()
        not_biases = list()

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endwith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        
        optimizer = torch.optim.SGD(params=[{'params':biases, 'lr':2 * lr}, {'params':not_biases}], lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']    
    
    