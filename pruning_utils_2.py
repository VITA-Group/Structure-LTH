import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

def need_to_prune(name, m, conv1):
    return ((name == 'conv1' and conv1) or (name != 'conv1')) \
        and isinstance(m, nn.Conv2d)

def pruning_model(model, px, conv1=True, random=False):

    print('start unstructured pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    parameters_to_prune.append((m,'weight'))
                else:
                    print('skip conv1 for L1 unstructure global pruning')
            else:
                parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    if not random:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    else:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=px,
        )

def prune_model_custom(model, mask_dict, conv1=False, random_index=-1, hold_sparsity = True):

    print('start unstructured pruning with custom mask')
    index = 0
    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):

            print("{}: {}".format(index, name))

            if index > random_index:
                print("origin: {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free: {}".format(index))
                number_of_zeros = (mask_dict[name+'.weight_mask'] == 0).sum()
                new_mask = torch.randn(mask_dict[name+'.weight_mask'].shape, device=mask_dict[name+'.weight_mask'].device)
                new_mask_2 = torch.randn(mask_dict[name+'.weight_mask'].shape, device=mask_dict[name+'.weight_mask'].device)
                threshold = np.sort(new_mask.view(-1).cpu().numpy())[number_of_zeros]
                new_mask_2[new_mask <= threshold] = 0
                new_mask_2[new_mask > threshold] = 1
                assert abs((new_mask_2 == 0).sum() - number_of_zeros) < 5 or (not hold_sparsity)
                assert (mask_dict[name+'.weight_mask'] - new_mask_2).abs().mean() > 0 # assert different mask
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2)
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())

            index += 1

def prune_model_custom_random(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index <= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    import random
    for i in range(50000):
        names_to_switch = np.random.choice(names, 2)
        name1 = names_to_switch[0]
        name2 = names_to_switch[1]
        limit = min(random_zeroes[name1], uppers[name2] - random_zeroes[name2])
        to_exchange = random.randint(0, limit)
        random_zeroes[name1] -= to_exchange
        random_zeroes[name2] += to_exchange

    print(random_zeroes)
    print(sum(random_zeroes.values()))
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index > random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                number_of_zeros = random_zeroes[name]
                new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = np.random.permutation(new_mask_2).reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=torch.from_numpy(new_mask_2).to(origin_mask.device))
                print((new_mask_2 == 0).sum() / new_mask_2.size)
            index += 1


def prune_model_custom_random_normal(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):
            if index <= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    
    number_of_zeros = sum(random_zeroes.values())
    number_of_elements = sum(uppers.values())

    random_zeroes = list(random_zeroes.values())
    uppers = list(uppers.values())
    indexes = [0]
    for i in range(len(random_zeroes)):
        indexes.append(sum(uppers[:(i+1)]))
    random_values = torch.randn(number_of_elements)
    threshold,_ = torch.topk(random_values, number_of_zeros)
    threshold = threshold[-1]

    new_masks_seq = torch.zeros(number_of_elements)
    new_masks_seq[random_values >= threshold] = 0
    new_masks_seq[random_values < threshold] = 1
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if need_to_prune(name, m, conv1):
            if index > random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                #number_of_zeros = random_zeroes[name]
                #new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = new_masks_seq[indexes[index]:indexes[index + 1]].reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2.to(origin_mask.device))
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())
            index += 1


def prune_model_custom_random_normal_reverse(model, mask_dict, conv1=True, random_index=-1):

    print('start unstructured pruning with custom mask')
    index = 0
    random_zeroes = {}
    zeroes = {}
    uppers = {}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index >= random_index:
                random_zeroes[name] = (mask_dict[name+'.weight_mask'] == 0).sum().item()
                uppers[name] = (mask_dict[name+'.weight_mask'].numel())
            
            index += 1
 
    print(random_zeroes)
    print(sum(random_zeroes.values()))
    names = list(random_zeroes.keys())
    print(uppers)
    
    number_of_zeros = sum(random_zeroes.values())
    number_of_elements = sum(uppers.values())

    random_zeroes = list(random_zeroes.values())
    uppers = list(uppers.values())
    indexes = [0]
    for i in range(len(random_zeroes)):
        indexes.append(sum(uppers[:(i+1)]))
    random_values = torch.randn(number_of_elements)
    threshold,_ = torch.topk(random_values, number_of_zeros)
    threshold = threshold[-1]

    new_masks_seq = torch.zeros(number_of_elements)
    new_masks_seq[random_values >= threshold] = 0
    new_masks_seq[random_values < threshold] = 1
    index = 0
    #random_zeros = {'conv1': 1708, 'layer1.0.conv1': 36492, 'layer1.0.conv2': 36502, 'layer1.1.conv1': 36505, 'layer1.1.conv2': 36500, 'layer2.0.conv1': 72973, 'layer2.0.conv2': 145958, 'layer2.0.downsample.0': 8108, 'layer2.1.conv1': 145978, 'layer2.1.conv2': 146033, 'layer3.0.conv1': 291894, 'layer3.0.conv2': 583861, 'layer3.0.downsample.0': 32439, 'layer3.1.conv1': 583925, 'layer3.1.conv2': 583984, 'layer4.0.conv1': 1167779, 'layer4.0.conv2': 2335680, 'layer4.0.downsample.0': 129812, 'layer4.1.conv1': 2335822, 'layer4.1.conv2': 2335687}
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if index < random_index:
                print("fix {}".format(index))
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print("free {}".format(index))
                origin_mask = mask_dict[name+'.weight_mask']
                #number_of_zeros = random_zeroes[name]
                #new_mask_2 = np.concatenate([np.zeros(number_of_zeros), np.ones(origin_mask.numel() - number_of_zeros)], 0)
                new_mask_2 = new_masks_seq[indexes[index - random_index]:indexes[index - random_index + 1]].reshape(origin_mask.shape)
        
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask_2.to(origin_mask.device))
                print((new_mask_2 == 0).sum().float() / new_mask_2.numel())
            index += 1

def remove_prune(model, conv1=True):
    
    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    prune.remove(m,'weight')
                else:
                    print('skip conv1 for remove pruning')
            else:
                prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def reverse_mask(mask_dict):
    new_dict = {}

    for key in mask_dict.keys():

        new_dict[key] = 1 - mask_dict[key]

    return new_dict

def extract_main_weight(model_dict, fc=True, conv1=True):
    new_dict = {}

    for key in model_dict.keys():
        if not 'mask' in key:
            if not 'normalize' in key:
                new_dict[key] = model_dict[key]

    if not fc:
        print('delete fc weight')

        delete_keys = []
        for key in new_dict.keys():
            if ('fc' in key) or ('classifier' in key):
                delete_keys.append(key)

        for key in delete_keys:
            del new_dict[key]

    if not conv1:
        print('delete conv1 weight')
        if 'conv1.weight' in new_dict.keys():
            del new_dict['conv1.weight']
        elif 'features.conv0.weight' in new_dict.keys():
            del new_dict['features.conv0.weight']
        elif 'conv1.0.weight' in new_dict.keys():
            del new_dict['conv1.0.weight']

    return new_dict

def check_sparsity(model, conv1=True):
    
    sum_list = 0
    zero_sum = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list+float(m.weight_mask.nelement())
                    zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))    
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list+float(m.weight_mask.nelement())
                zero_sum = zero_sum+float(torch.sum(m.weight_mask == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def mask_add_back(mask_dict):

    new_mask_dict = {}
    rate_list = []

    for key in mask_dict.keys():

        shape_0 = mask_dict[key].size(0)
        reshape_mask = mask_dict[key].reshape(shape_0, -1)
        zero_number = torch.mean(reshape_mask.eq(0).float(), dim=1)
        rate_list.append(zero_number)

        new_mask = torch.zeros_like(mask_dict[key])
        for indx in range(shape_0):
            if zero_number[indx] != 1:
                new_mask[indx,:] = 1

        new_mask_dict[key] = new_mask

    rate_list = torch.cat(rate_list, dim=0)
    print('all_channels: ', rate_list.shape)
    print('full zero channels: ', torch.sum(rate_list.eq(1).float()))

    return new_mask_dict

def check_zero_channel(mask_dict):

    rate_list = []

    for key in mask_dict.keys():

        shape_0 = mask_dict[key].size(0)
        reshape_mask = mask_dict[key].reshape(shape_0, -1)
        zero_number = torch.mean(reshape_mask.eq(0).float(), dim=1)
        rate_list.append(zero_number)

    rate_list = torch.cat(rate_list, dim=0)
    all_channels_number = rate_list.shape[0]
    zero_channels_number = torch.sum(rate_list.eq(1).float()).item()
    zero_channel_rate = 100*zero_channels_number/all_channels_number 

    print('all_channels: ', all_channels_number)
    print('full zero channels: ', zero_channels_number)
    print('* zero channels rate: {}% '.format(zero_channel_rate))

    return zero_channel_rate




def prune_model_custom_one_random(model, mask_dict, random_index = -1):

    index = 0
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):

            print('pruning layer with custom mask:', name)
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            if index == random_index:
                prune.RandomUnstructured.apply(m, 'weight', amount=(mask_dict[name+'.weight_mask']==0).sum().int().item() / mask_dict[name+'.weight_mask'].numel())
            index += 1