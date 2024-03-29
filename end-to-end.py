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
from numpy.lib.arraysetops import isin

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
from advertorch.utils import NormalizeByChannelMeanStd

from utils import *
from pruning_utils_2 import *
from pruning_utils_unprune import *
from pruning_utils import prune_model_custom_fillback
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
parser.add_argument('--reuse', action="store_true")
parser.add_argument('--use-original', action="store_true")
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--fillback-rate', type=float)



best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    args.use_sparse_conv = False
    args.batch_size=32
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
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    except:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    start = time.time()
    current_mask = extract_mask(state_dict)
    print(current_mask.keys())
    
    from models.conv import SparseConv2D
    
    combined_state_dict = {}
    for key in state_dict:
        if key in current_mask:
            combined_state_dict[key[:-5]] = current_mask[key] * state_dict[key[:-5] + "_orig"]
        elif not 'orig' in key:
            combined_state_dict[key] = state_dict[key]
    
    model.load_state_dict(combined_state_dict, strict=False)
    model = model.cuda()
    from models.conv import SparseConv2D
    
    def replace_conv(m, name):
        print(name)
        for attr_str, _ in m.named_children():
            print(attr_str)
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, nn.Conv2d) and not args.use_original:
                record = copy.deepcopy(getattr(m, attr_str))
                new_conv = SparseConv2D(target_attr.weight.shape[1], target_attr.weight.shape[0], target_attr.weight.shape[2], target_attr.stride, target_attr.padding, target_attr.dilation, False, reuse=args.reuse, identifier=attr_str)
                flag = new_conv.load(record.weight.data.detach(), None)
                if (flag > 0):
                    setattr(m, attr_str, new_conv)
                    print(f"DENSE BLOCKS GREATER THAN 0 in {name}")
                else:
                    print(f"NO DENSE BLOCK WAS FOUND in {name}")
            replace_conv(_, attr_str)
    replace_conv(model, "new_model")
    
    import torchprof
    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        for i in range(10):
            with torch.no_grad():
                x = torch.randn((64, 3, 32, 32)).cuda()
                output = model(x)
                del output
    
    print(prof.display(show_events=False))
    # x = torch.randn((64, 3, 32, 32)).cuda()
    # output = model(x)
def save_checkpoint(state, is_SA_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))

def load_ticket(model, args):
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


