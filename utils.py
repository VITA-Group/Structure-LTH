import copy 
import torch
import numpy as np
from torchvision.models import mobilenet
from models.resnet import resnet18, resnet50, resnet152, resnext101_32x8d, wide_resnet50_2, resnext50_32x4d, resnet18_2, resnet18_3, resnet18_4, resnet18_5, resnet18_6, resnet50_2, resnet50_3, resnet50_4, resnet50_5, resnet50_6
from models.resnets import resnet20, resnet56
from models.densenet import densenet161
from models.shufflenet import shufflenet_v2_x1_0
from models.resnets_2fc import resnet20 as resnet20_2fc
from models.mobilenet import MobileNet, MobileNet1, MobileNet2, MobileNet4, MobileNet3, MobileNet5
from models.resnet12 import resnet12
from advertorch.utils import NormalizeByChannelMeanStd
from dataset import *
from models.vgg import vgg16_bn
from models.resnet_grasp import resnet32

def setup_model_dataset(args):
    trigger_set_dataloader = None
    if args.dataset == 'cifar10':
        classes = 10
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    
    elif args.dataset == 'cifar10_trigger':
        classes = 10
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader, trigger_set_dataloader = cifar10_with_trigger_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    elif args.dataset == 'cifar100_trigger':
        classes = 100
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_set_loader, val_loader, test_loader, trigger_set_dataloader = cifar100_with_trigger_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    elif args.dataset == 'cifar100':
        classes = 100
        train_number = 45000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size= args.batch_size, data_dir =args.data)

    elif args.dataset == 'tiny-imagenet':
        classes = 200
        train_number = 90000
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        train_set_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size= args.batch_size, data_dir =args.data)
    else:
        raise ValueError('unknow dataset')

    if args.arch == 'res18':
        print('build model resnet18')
        model = resnet18(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet', use_sparse_conv=args.use_sparse_conv)
    elif args.arch == 'res18_2':
        print('build model resnet18-2')
        model = resnet18_2(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res18_3':
        print('build model resnet18-3')
        model = resnet18_3(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res18_4':
        print('build model resnet18-4')
        model = resnet18_4(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res18_5':
        print('build model resnet18-5')
        model = resnet18_5(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res18_6':
        print('build model resnet18-6')
        model = resnet18_6(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res50':
        print('build model resnet50')
        model = resnet50(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet', use_sparse_conv=args.use_sparse_conv)
    elif args.arch == 'res50_2':
        print('build model resnet50-2')
        model = resnet50_2(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res50_3':
        print('build model resnet50-3')
        model = resnet50_3(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res50_4':
        print('build model resnet50-4')
        model = resnet50_4(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res50_5':
        print('build model resnet50-5')
        model = resnet50_5(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res50_6':
        print('build model resnet50-6')
        model = resnet50_6(num_classes=classes, imagenet=args.dataset == 'tiny-imagenet')
    elif args.arch == 'res20s':
        print('build model: resnet20')
        model = resnet20(number_class=classes)
    elif args.arch == 'res56s':
        print('build model: resnet56')
        model = resnet56(number_class=classes)
    elif args.arch == 'vgg16_bn':
        print('build model: vgg16_bn')
        model = vgg16_bn(num_classes=classes, use_sparse_conv=args.use_sparse_conv)
    elif args.arch == 'mobilenet':
        print('build model: mobilenet')
        model = MobileNet(num_classes=classes)
    elif args.arch == 'mobilenet1':
        print('build model: mobilenet1')
        model = MobileNet1(num_classes=classes)
    elif args.arch == 'mobilenet2':
        print('build model: mobilenet2')
        model = MobileNet2(num_classes=classes)
    elif args.arch == 'mobilenet3':
        print('build model: mobilenet3')
        model = MobileNet3(num_classes=classes)
    elif args.arch == 'mobilenet4':
        print('build model: mobilenet4')
        model = MobileNet4(num_classes=classes)
    elif args.arch == 'mobilenet5':
        print('build model: mobilenet5')
        model = MobileNet5(num_classes=classes)
    elif args.arch == 'resnet12':
        print('build model: resnet12')
        model = resnet12(num_classes=classes)
    elif args.arch == 'resnet32_grasp':
        model = resnet32(dataset=args.dataset)

    else:
        raise ValueError('unknow model')

    model.normalize = normalization
    if not trigger_set_dataloader is None:
        return model, train_set_loader, val_loader, test_loader, trigger_set_dataloader
    else:
        return model, train_set_loader, val_loader, test_loader


def cvt_state_dict(state_dict, adv_simclr, bn_idx=0):

    state_dict_new = copy.deepcopy(state_dict)

    if adv_simclr:

        for name, item in state_dict.items():

            if 'downsample.conv' in name:
                state_dict_new[name.replace('downsample.conv', 'downsample.0')] = item
                del state_dict_new[name]
                continue

            if 'downsample.bn' in name:
                state_dict_new[name.replace('downsample.bn.bn_list.'+str(bn_idx), 'downsample.1')] = item
                del state_dict_new[name]
                continue
            
            if not 'fc' in name:

                if 'bn_list.'+str(bn_idx) in name:
                    state_dict_new[name.replace('.bn_list.'+str(bn_idx), '')] = item
                    del state_dict_new[name]
                    continue

                if 'bn_list.'+str(1-bn_idx) in name:
                    del state_dict_new[name]
                    continue

            else:
                del state_dict_new[name]

    else:

        for name, item in state_dict.items():

            if 'downsample.conv' in name:
                state_dict_new[name.replace('downsample.conv', 'downsample.0')] = item
                del state_dict_new[name]
                continue

            if 'downsample.bn' in name:
                state_dict_new[name.replace('downsample.bn', 'downsample.1')] = item
                del state_dict_new[name]
                continue

            if 'fc' in name:
                del state_dict_new[name]

    new_dict = {}
    for key in state_dict_new.keys():

        if 'module' in key:
            new_key = key[len('module.'):]
        else:
            new_key = key

        new_dict[new_key] = state_dict_new[key]

    return new_dict

def moco_state_dict(state_dict):
    new_dict = {}

    for key in state_dict.keys():
        if 'module.encoder_q.' in key:
            new_key = key[len('module.encoder_q.'):]
            if not 'fc' in key:
                new_dict[new_key] = state_dict[key]

    return new_dict

def forget_times(record_list):
    
    number = 0
    learned = False

    for i in range(record_list.shape[0]):
        
        if not learned:
            if record_list[i] == 1:
                learned = True 

        else:
            if record_list[i] == 0:
                learned = False
                number+=1 

    return number

def sorted_examples(example_wise_prediction, data_prune, data_rate, state):

    forgetting_events_number = np.zeros(example_wise_prediction.shape[0])
    for j in range(example_wise_prediction.shape[0]):
        tmp_data = example_wise_prediction[j,:]
        if tmp_data[0] < 0:
            forgetting_events_number[j] = -1 
        else:
            forgetting_events_number[j] = forget_times(tmp_data)
    
    if data_prune == 'constent':
        print('* pruning {} data'.format(data_rate))
        rest_number = int(45000*(1-data_rate)**state)
    elif data_prune == 'zero_out':
        print('zero all unforgettable images out')
        rest_number = np.where(forgetting_events_number>0)[0].shape[0]
    else:
        print('error data_prune type')
        assert False

    sequence = np.argsort(forgetting_events_number)[-rest_number:]

    return sequence

def split_class_sequence(sequence, all_labels, num_class):
    
    class_wise_sequence = {}
    for i in range(num_class):
        class_wise_sequence[i] = []
    
    for index in range(sequence.shape[0]):
        class_wise_sequence[all_labels[sequence[index]]].append(sequence[index])
    
    for i in range(num_class):
        class_wise_sequence[i] = np.array(class_wise_sequence[i])
        print('class = {0}, number = {1}'.format(i, class_wise_sequence[i].shape[0]))

    return class_wise_sequence

def blance_dataset_sequence(class_wise_sequence, num_class):
    class_wise_number = np.zeros(num_class, dtype=np.int)
    for i in range(num_class):
        class_wise_number[i] = class_wise_sequence[i].shape[0]
    
    max_length = np.max(class_wise_number)
    print('max class number = {}'.format(max_length))

    balance_sequence = []
    arange_max = np.arange(max_length)
    for i in range(num_class):

        shuffle_index = np.random.permutation(class_wise_number[i])
        shuffle_class_sequence = class_wise_sequence[i][shuffle_index]
        balance_sequence.append(shuffle_class_sequence[arange_max%class_wise_number[i]])

    balance_sequence = np.concatenate(balance_sequence)
    print(balance_sequence.shape)
    return balance_sequence
