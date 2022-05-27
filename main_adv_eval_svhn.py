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
from dataset import cifar100_dataloaders, cifar10_dataloaders, svhn_dataloaders, cifar10_dataloaders_2, cifar10_dataloaders_3

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### General Setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='svhn', help='dataset')
parser.add_argument('--data_number', default=1.0, type=float, help='number of images for training [0,1], only support for CIFAR')
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--init', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--save_all', action="store_true", help="save checkpoint after each epoch")

##################################### Training Setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='50,75', help='decreasing strategy')

##################################### Attack Setting #################################################
parser.add_argument('--train_eps', default=(8/255), type=float, help='train_eps')
parser.add_argument('--train_step', default=10, type=int, help='train_steps')
parser.add_argument('--train_gamma', default=(2/255), type=float, help='train_gamma')
parser.add_argument('--test_eps', default=(8/255), type=float, help='train_eps')
parser.add_argument('--test_step', default=20, type=int, help='train_steps')
parser.add_argument('--test_gamma', default=(2/255), type=float, help='train_gamma')

##################################### Pruning Setting #################################################
parser.add_argument('--tickets_file', default=None, type=str, help='tickets file')

best_sa = 0
best_ra = 0

def main():
    global args, best_sa, best_ra
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    #load initialization
    if args.init:
        pretrained_init = torch.load(args.init, map_location='cpu')
        print("=> loading weight from '{}'".format(args.init))
        model.load_state_dict(pretrained_init)

    # cifar100
    if args.dataset == 'cifar100':
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size=args.batch_size, data_dir= args.data, data_rate=args.data_number, add_normalize=False)
        classes=100
        normalize_layer = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        print('* dataset: cifar100')

    elif args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size=args.batch_size, data_dir= args.data, data_rate=args.data_number, add_normalize=False)
        classes=10
        normalize_layer = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        print('* dataset: cifar10')

    elif args.dataset == 'svhn':
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size=args.batch_size, data_dir= args.data, data_rate=args.data_number, add_normalize=False)
        classes=10
        normalize_layer = NormalizeByChannelMeanStd(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        print('* dataset: svhn')

    elif args.dataset == 'cifar10_2':
        train_loader, val_loader, test_loader = cifar10_dataloaders_2(batch_size=args.batch_size, data_dir= args.data, data_rate=args.data_number, add_normalize=False)
        classes=10
        normalize_layer = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        print('* dataset: cifar10_2')
    elif args.dataset == 'cifar10_3':
        train_loader, val_loader, test_loader = cifar10_dataloaders_3(batch_size=args.batch_size, data_dir= args.data, data_rate=args.data_number, add_normalize=False)
        classes=10
        normalize_layer = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        print('* dataset: cifar10_3')

    if 'vgg' in args.arch:
        features_number = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(features_number, classes)       

    else:
        features_number = model.fc.in_features
        model.fc = nn.Linear(features_number, classes)
        if args.dataset == 'cifar10_3':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

    #load mask 
    if args.tickets_file:
        mask_file = torch.load(args.tickets_file, map_location='cpu')
        if 'state_dict' in mask_file.keys():
            mask_file = mask_file['state_dict']
        current_mask = extract_mask(mask_file)
        prune_model_custom(model, current_mask)

    #add normalize layer 
    model = nn.Sequential(normalize_layer, model)
    # print(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))
        
        best_sa = checkpoint['best_sa']
        best_ra = checkpoint['best_ra']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('loading from epoch: ', start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []
        all_result['test_ra'] = []
        all_result['ra'] = []

        start_epoch = 0

    check_sparsity(model)        
    for epoch in range(start_epoch, args.epochs):

        start = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion)
        # robust evaluate on validation set
        atacc = validate_adv(val_loader, model, criterion)

        scheduler.step()

        sa_best = tacc > best_sa
        best_sa = max(tacc, best_sa)

        ra_best = atacc > best_ra
        best_ra = max(atacc, best_ra)

        all_result['train'].append(acc)
        all_result['ta'].append(tacc)
        all_result['ra'].append(atacc)        
        all_result['test_ta'].append(-1)
        all_result['test_ra'].append(-1)

        if args.save_all:
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}-checkpoint.pth.tar'.format(epoch)))

        save_checkpoint({
            'best_sa': best_sa,
            'best_ra': best_ra,
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, sa_best, ra_best, save_path=args.save_dir)
    
        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['test_ra'], label='robust_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

        end = time.time()
        print('* Time = {}'.format(end-start))

    #report result
    check_sparsity(model)

    # TA best 
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_SA_best.pth.tar'))['state_dict'])
    test_tacc = validate(test_loader, model, criterion)
    test_atacc = validate_adv(test_loader, model, criterion)
    print('* TA best SA={}'.format(test_tacc))
    print('* TA best RA={}'.format(test_atacc))

    # RA best 
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_RA_best.pth.tar'))['state_dict'])
    test_tacc = validate(test_loader, model, criterion)
    test_atacc = validate_adv(test_loader, model, criterion)
    print('* RA best SA={}'.format(test_tacc))
    print('* RA best RA={}'.format(test_atacc))


def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    adversary = LinfPGDAttack(
            model, 
            loss_fn=F.cross_entropy, 
            eps=args.train_eps, 
            nb_iter=args.train_step, 
            eps_iter=args.train_gamma,
            rand_init=True, 
            clip_min=0.0, clip_max=1.0, targeted=False
        )

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        with ctx_noparamgrad(model):
            images_adv = adversary.perturb(image, target)

        # compute output
        output_adv = model(images_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
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

def validate_adv(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    adversary = LinfPGDAttack(
            model, 
            loss_fn=F.cross_entropy, 
            eps=args.test_eps, 
            nb_iter=args.test_step, 
            eps_iter=args.test_gamma,
            rand_init=True,
            clip_min=0.0, clip_max=1.0, targeted=False
        )

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        images_adv = adversary.perturb(image, target)

        # compute output
        with torch.no_grad():
            output = model(images_adv)
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

def save_checkpoint(state, ta_best, ra_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if ta_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if ra_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))



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


