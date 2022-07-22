'''
iterative pruning for supervised task 
with lottery tickets or pretrain tickets 
support datasets: cifar10, Fashionmnist, cifar100
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

from utils import *
from pruning_utils import *
from model_utils import accuracy, setup_seed, warmup_lr, AverageMeter, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Iterative Pruning')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='res20s', help='model architecture')
parser.add_argument('--file_name', type=str, default=None, help='dataset index')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--init', type=str, default=None, help='init file')

##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt,pt or pt_trans)')
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
parser.add_argument('--fc', action="store_true", help="whether rewind fc")
parser.add_argument('--rewind_epoch', default=2, type=int, help='rewind checkpoint')


best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    args.use_sparse_conv = False
    print(args)

    print('*'*50)
    print('conv1 included for prune and rewind: {}'.format(args.conv1))
    print('fc included for rewind: {}'.format(args.fc))
    print('*'*50)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    print(model.normalize)  
    new_initialization = copy.deepcopy(model.state_dict())
    if not os.path.exists(args.init):
        torch.save(new_initialization, args.init)
    initialization = torch.load(args.init)
    if 'state_dict' in initialization:
        initialization = initialization['state_dict']
    
    initialization['normalize.mean'] = new_initialization['normalize.mean']
    initialization['normalize.std'] = new_initialization['normalize.std']

    print(initialization.keys())
    if not args.prune_type == 'lt':
        keys = list(initialization.keys())
        for key in keys:
            if key.startswith('fc') or key.startswith('conv1'):
                del initialization[key]

        initialization['fc.weight'] = new_initialization['fc.weight']
        initialization['fc.bias'] = new_initialization['fc.bias']
        initialization['conv1.weight'] = new_initialization['conv1.weight']
        model.load_state_dict(initialization)
    else:
        print(initialization.keys())
        model.load_state_dict(initialization)
        
    if args.resume:
        print('resume from checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        try:
            best_sa = checkpoint['best_sa']
            all_result = checkpoint['result']
        except:
            best_sa = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        start_state = checkpoint['state'] if 'state' in checkpoint else 1

        if start_state>0:
            current_mask = extract_mask(checkpoint['state_dict'])
            for m in current_mask:
                print(current_mask[m].float().mean())
            #print(current_mask)
            prune_model_custom(model, current_mask)
            check_sparsity(model, conv1=False)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
        if 'normalize.mean' not in checkpoint['state_dict']:
            checkpoint['state_dict']['normalize.mean'] = new_initialization['normalize.mean']
            checkpoint['state_dict']['normalize.std'] = new_initialization['normalize.std']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initialization = copy.deepcopy(checkpoint['init_weight']) if 'init_weight' in checkpoint else torch.load(args.init)['state_dict']

        if 'normalize.mean' not in initialization:
            initialization['normalize.mean'] = new_initialization['normalize.mean']
            initialization['normalize.std'] = new_initialization['normalize.std']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)
        all_result = {}
        all_result['train'] = [best_sa]
        all_result['test_ta'] = [best_sa]
        all_result['ta'] = [best_sa]

    else:
        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        start_epoch = 0
        start_state = 0

    print('######################################## Start Standard Training Iterative Pruning ########################################')
    
    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        
        check_sparsity(model, conv1=False)        
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if epoch == args.rewind_epoch and args.prune_type == 'rewind_lt' and state == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{args.rewind_epoch}.pth.tar"))
                initialization = copy.deepcopy(model.state_dict())
            acc = train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            tacc = validate(val_loader, model, criterion)
            # evaluate on test set
            test_tacc = validate(test_loader, model, criterion)

            scheduler.step()

            all_result['train'].append(acc)
            all_result['ta'].append(tacc)
            all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc  > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initialization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        #report result
        validate(val_loader, model, criterion) # extra forward
        check_sparsity(model, conv1=False)
        print('* best SA={}'.format(all_result['test_ta'][np.argmax(np.array(all_result['ta']))]))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0

        pruning_model(model, args.rate, conv1=False)
        remain_weight = check_sparsity(model, conv1=False)
        current_mask = extract_mask(model.state_dict())
        for m in current_mask:
            print(current_mask[m].float().mean())

        remove_prune(model, conv1=False)

        model.load_state_dict(initialization)
        prune_model_custom_fillback(model, current_mask, train_loader=train_loader)
        check_sparsity(model, conv1=False)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)


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


def load_weight(model, initialization, args): 
    print('loading pretrained weight')
    loading_weight = extract_main_weight(initialization)
    
    for key in loading_weight.keys():
        if not (key in model.state_dict().keys()):
            print(key)
            assert False

    print('*number of loading weight={}'.format(len(loading_weight.keys())))
    print('*number of model weight={}'.format(len(model.state_dict().keys())))
    model.load_state_dict(loading_weight)


if __name__ == '__main__':
    main()


