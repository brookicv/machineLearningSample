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
workers = 4
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

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.prior_cxcy).to(device)

    train_dataset = PascalVocDataset(data_folder,split='train',keep_difficult=keep_diffcult)
    val_dataset = PascalVocDataset(data_folder,split='test',keep_diffcult=keep_diffcult)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=workers,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True,collate_fn=val_dataset.collate_fn,num_workers=workers,pin_memory=True)

    # epochs
    for epoch in range(start_epoch,epochs):
        
        # One epoch training
        train(train_loader=train_loader,model=model,criterion=criterion,optimizer=optimizer,epoch=epoch)

        # One epoch validation
        val_loss = validate(val_loader=val_loader,model=model,criterion=criterion)

        is_best = val_loss < best_loss
        best_loss = min(val_loss,best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\n Epochs since last improvement:%d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        save_checkpoint(epoch,epochs_since_improvement,model,optimizer,val_loss,best_loss,is_best)

def train(train_loader,model,criterion,optimizer,epoch):

    model.train() # enable dropout

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i,(images,boxes,labels,_) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) fro l in labels]

        # Forward prop.
        predicted_locs,predicted_scores = model(images) # (n,8732,4),(n,8732,n_classes)

        loss = criterion(predicted_locs,predicted_scores,boxes,labels)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)

        # update model
        optimizer.step()

        losses.update(loss.item(),images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

         # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def validate(val_loader,model,criterion):

    model.eval() # eval mode dsables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for i,(images,boxes,labels,difficulties) in enumerate(val_loader):

            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs,predicted_scores = model(images)

            loss = criterion(predicted_locs,predicted_scores,boxes,labels)

            losses.update(loss.item(),images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
        
        print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

        return losses.avg

if if __name__ == "__main__":
    main()
    
    
    