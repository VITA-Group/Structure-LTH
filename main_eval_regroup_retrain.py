'''
load lottery tickets and evaluation 
support datasets: cifar10, Fashionmnist, cifar100
'''

import os
import time 
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
from pruning_utils import regroup
from pruning_utils_2 import *
from pruning_utils import prune_model_custom_fillback
from model_utils import accuracy, setup_seed, warmup_lr, AverageMeter, save_checkpoint
parser = argparse.ArgumentParser(description='PyTorch Evaluation Tickets')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch', type=str, default='res18', help='model architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save_model', action="store_true", help="whether saving model")

##################################### training setting #################################################
parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pretrained', default=None, type=str, help='pretrained weight for pt')
parser.add_argument('--mask_dir', default=None, type=str, help='mask direction for ticket')
parser.add_argument('--conv1', action="store_true", help="whether pruning&rewind conv1")
parser.add_argument('--fc', action="store_true", help="whether rewind fc")

parser.add_argument('--type', type=str, default=None, choices=['ewp', 'random_path', 'betweenness', 'hessian_abs', 'taylor1_abs','intgrads','identity', 'omp'])
parser.add_argument('--add-back', action="store_true", help="add back weights")
parser.add_argument('--prune-type', type=str, choices=["lt", 'pt', 'st', 'mt', 'trained', 'transfer'])
parser.add_argument('--num-paths', default=50000, type=int)
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--evaluate-p', type=float, default=0.00)
parser.add_argument('--evaluate-random', action="store_true")
parser.add_argument('--evaluate-full', action="store_true")

parser.add_argument('--checkpoint', type=str)
parser.add_argument('--criterions', default="remain", type=str, choices=['remain', 'magnitude', 'l1', 'l2', 'taylor'])

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
    
    criterion = nn.CrossEntropyLoss()
    model.cuda()

    load_ticket(model, args, train_loader=train_loader)

    
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
    if args.mask_dir:
        remain_weight = check_sparsity(model, conv1=args.conv1)

    for epoch in range(start_epoch, args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion)

        scheduler.step()

        all_result['train'].append(acc)
        all_result['ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)
        all_result['remain_weight'] = remain_weight

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc  > best_sa
        best_sa = max(tacc, best_sa)

        if args.save_model:

            save_checkpoint({
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_SA_best=is_best_sa, save_path=args.save_dir)

        else:
            save_checkpoint({
                'result': all_result
            }, is_SA_best=False, save_path=args.save_dir)

        plt.plot(all_result['train'], label='train_acc')
        plt.plot(all_result['ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    check_sparsity(model, conv1=args.conv1)
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


def load_ticket(model, args, train_loader=None):
    # weight 
    if args.pretrained:

        initalization = torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))
        
        if 'init_weight' in initalization.keys():
            print('loading from init_weight')
            initalization = initalization['init_weight']
        elif 'state_dict' in initalization.keys():
            print('loading from state_dict')
            initalization = initalization['state_dict']
        
        loading_weight = extract_main_weight(initalization, fc=True, conv1=True)
        new_initialization = model.state_dict()
        if not 'normalize.std' in loading_weight:
            loading_weight['normalize.std'] = new_initialization['normalize.std']
            loading_weight['normalize.mean'] = new_initialization['normalize.mean']

        if not (args.prune_type == 'lt' or args.prune_type == 'trained'):
            keys = list(loading_weight.keys()) 
            for key in keys:
                if key.startswith('fc') or key.startswith('conv1'):
                    del loading_weight[key]

            loading_weight['fc.weight'] = new_initialization['fc.weight']
            loading_weight['fc.bias'] = new_initialization['fc.bias']
            loading_weight['conv1.weight'] = new_initialization['conv1.weight']

        print('*number of loading weight={}'.format(len(loading_weight.keys())))
        print('*number of model weight={}'.format(len(model.state_dict().keys())))
        model.load_state_dict(loading_weight)
        


    # mask 
    if args.mask_dir:
        print('loading mask')
        current_mask_weight = torch.load(args.mask_dir, map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in current_mask_weight.keys():
            current_mask_weight = current_mask_weight['state_dict']
        current_mask = extract_mask(current_mask_weight)

        checkpoint =  torch.load(args.pretrained, map_location = torch.device('cuda:'+str(args.gpu)))

        for key in current_mask:
            mask = current_mask[key]
            shape = current_mask[key].shape
            current_mask[key] = regroup(mask.view(shape[0], -1)).view(*shape)
            print(current_mask[key].mean())
        prune_model_custom(model, current_mask)
        #prune_random_betweeness(model, current_mask, int(args.num_paths), downsample=downsample, conv1=args.conv1)
        check_sparsity(model, conv1=args.conv1)


if __name__ == '__main__':
    main()


