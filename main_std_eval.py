'''
transfer learning of lottery tickets
'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from utils_pruning import *
from dataset import cifar100_dataloaders, cifar10_dataloaders, svhn_dataloaders, visda_loader

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### General Setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--init', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

##################################### Training Setting #################################################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='50,100', help='decreasing strategy')

##################################### Pruning Setting #################################################
parser.add_argument('--tickets_file', default=None, type=str, help='tickets file')

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    #load initialization
    pretrained_init = torch.load(args.init, map_location='cpu')
    print("=> loading weight from '{}'".format(args.init))
    model.load_state_dict(pretrained_init)

    # cifar100
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size=args.batch_size, data_dir= args.data)
        classes=100
        print('* dataset: cifar100')
    elif args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size=args.batch_size, data_dir= args.data)
        classes=10
        print('* dataset: cifar10')
    elif args.dataset == 'svhn':
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size=args.batch_size, data_dir= args.data)
        classes=10
        print('* dataset: svhn')
    elif args.dataset == 'visda':
        train_loader, val_loader, test_loader = visda_loader(batch_size=args.batch_size, data_dir= args.data)
        classes=12
        print('* dataset: VisDA-2017')


    features_number = model.fc.in_features
    model.fc = nn.Linear(features_number, classes)

    #load mask 
    if args.tickets_file:
        mask_file = torch.load(args.tickets_file, map_location='cpu')
        if 'state_dict' in mask_file.keys():
            mask_file = mask_file['state_dict']
        current_mask = extract_mask(mask_file)
        prune_model_custom(model, current_mask)

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    all_result = {}
    all_result['train'] = []
    all_result['test_ta'] = []
    all_result['ta'] = []

    start_epoch = 0

    check_sparsity(model)        
    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion)
        # evaluate on test set
        if args.dataset == 'visda':
            test_tacc = tacc
        else:
            test_tacc = validate(test_loader, model, criterion)

        scheduler.step()

        all_result['train'].append(acc)
        all_result['ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint({
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }, is_best_sa, save_path=args.save_dir)
    
        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    #report result
    check_sparsity(model)
    print('* last SA={}'.format(all_result['test_ta'][-1]))
    print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def save_checkpoint(state, ta_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if ta_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()


