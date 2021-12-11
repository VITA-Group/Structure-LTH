import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from advertorch.utils import NormalizeByChannelMeanStd
from .channel_selection import channel_selection
# from utils.common_utils import try_cuda
# from .init_utils import weights_init

__all__ = ['resnet']  # , 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
_AFFINE = True
#_AFFINE = False


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, cfg, stride=1):
    
        super(BasicBlock, self).__init__()
        if (cfg[0] < in_planes):
            self.select = channel_selection(in_planes)
            self.select.indexes.data = torch.cat([torch.ones(cfg[0]), torch.zeros(in_planes - cfg[0])], 0)
        else:
            self.select = None
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1], affine=_AFFINE)
        self.conv2 = nn.Conv2d(cfg[1], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)

        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(cfg[0], self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        if self.select is not None:
            out = self.select(x)
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cfg=None):

        super(ResNet, self).__init__()
        if cfg is None:
            cfg = [32]  + [32, 32] *  num_blocks[0] + [32, 64]  + [64, 64] * (num_blocks[1] - 1) + [64, 128] + [128, 128] * (num_blocks[2] - 1)
        _outputs = [32, 64, 128]
        self.in_planes = cfg[0]

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, 32, cfg[1: 1 + 2*num_blocks[0]], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, cfg[1 + 2*num_blocks[0]:1 + 2*(num_blocks[0] + num_blocks[1])], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, cfg[1 + 2*(num_blocks[0] + num_blocks[1]):], num_blocks[2], stride=2)
        self.linear = nn.Linear(128, num_classes)
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.apply(weights_init)

    def _make_layer(self, block, planes, cfgs, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, cfgs[2*i:2*(i+1)], stride))
            i += 1
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(depth=32, dataset='cifar10', cfg=None):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth
    n = (depth - 2) // 6
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    else:
        raise NotImplementedError('Dataset [%s] is not supported.' % dataset)
    return ResNet(BasicBlock, [n]*3, num_classes, cfg=cfg)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


def weights_init(m):
    # print('=> weights init')
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # nn.init.xavier_normal(m.weight)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are
        # already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def resnet32(dataset):
    return resnet(depth=32, dataset=dataset)
