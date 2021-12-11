from pickle import TRUE
import torch
import numpy as np
import torch.nn as nn
from models.resnet_cfg import *
from models.channel_selection import channel_selection
import numpy as np



def prune_model_custom_one_shot_channel(model, percent, conv1=False, init_weight=None, trained_weight=None):
    try:
        model.load_state_dict(trained_weight)
    except:
        for key in list(trained_weight.keys()):
            if ('mask' in key):
                trained_weight[key[:-5]] = trained_weight[key[:-5] + "_orig"] * trained_weight[key]
                del trained_weight[key[:-5] + "_orig"]
                del trained_weight[key]
        model.load_state_dict(trained_weight, strict=False)

    total = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and not 'downsample' in name:
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and not 'downsample' in name:
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * percent)
    thre = y[thre_index]


    pruned = 0
    cfg = []
    cfg_mask = []
    for k, (name,m) in enumerate(model.named_modules()):
        if isinstance(m, nn.BatchNorm2d) and not 'downsample' in name:
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    pruned_ratio = pruned / total
    p_flops += 2 * total #(minus and sum)

    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    print('Pre-processing Successful!')

    print("Cfg:")
    print(cfg)


    newmodel = resnet18(num_classes=10, cfg=cfg)


    model.load_state_dict(init_weight, strict=False)

    newmodel.to(next(model.parameters()).device)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())

    useful_i = []
    for i, module in enumerate(old_modules):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
            useful_i.append(i)
    temp = []
    for i, item in enumerate(useful_i):
        temp.append(old_modules[item])

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0


    downsample = [8, 13, 18]
    last_block = [3, 5, 7, 10, 12, 15, 17, 20]
    for layer_id in range(len(temp)):
        m0 = old_modules[useful_i[layer_id]]
        m1 = new_modules[useful_i[layer_id]]
        # print(m0)
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(layer_id_in_cfg, len(cfg_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[useful_i[layer_id+2]] # channel selection
                assert isinstance(m2, channel_selection)
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif bn_count in downsample:
                # If the current layer is the downsample layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

            elif bn_count in last_block:
                # If the current layer is the last conv-bn layer in block, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                if bn_count + 1 in downsample:
                    m2 = new_modules[useful_i[layer_id+3]]
                    assert isinstance(m2, channel_selection)
                else:
                    m2 = new_modules[useful_i[layer_id+1]]
                    assert isinstance(m2, channel_selection) or isinstance(m2, nn.Linear)
                if isinstance(m2, channel_selection):
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in downsample: # downsample
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in last_block:
                # the last convolution in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # end_mask = cfg_mask[-1]
            # idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx0)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    model = newmodel
    return model

def prune_model_custom_channel(model, percent, conv1=False, init_weight=None, trained_weight=None, train_loader=False, criterion="weight"):
    
    try:
        model.load_state_dict(trained_weight)
    except:
        for key in list(trained_weight.keys()):
            if ('mask' in key):
                trained_weight[key[:-5]] = trained_weight[key[:-5] + "_orig"] * trained_weight[key]
                del trained_weight[key[:-5] + "_orig"]
                del trained_weight[key]
        model.load_state_dict(trained_weight, strict=False)

    feature_maps = []
    def hook(module, input, output):
        feature_maps.append(output)
    
    image, label = next(iter(train_loader))
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
            device = m.weight.data.device
    output = model(image.to(device))
    loss = torch.nn.CrossEntropyLoss()(output, label.cuda())
    total = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    counter = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.shape[0]
            if criterion == 'weight':
                bn[index:(index+size)] = m.weight.data.abs().mean((1,2,3)).clone()
            elif criterion == 'l1':
                bn[index:(index+size)] = feature_maps[counter].abs().mean((0,2,3)).clone()
                counter += 1
            elif criterion == 'l2':
                bn[index:(index+size)] = (feature_maps[counter].abs() ** 2).mean((0,2,3)).clone()
                counter += 1
            elif criterion == 'saliency':
                bn[index:(index+size)] = (feature_maps[counter] * torch.autograd.grad(loss, feature_maps[counter], only_inputs=True, retain_graph=True)[0].detach()).abs().mean((0,2,3)).clone()
                counter += 1
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * percent)
    thre = y[thre_index]


    pruned = 0
    cfg = []
    cfg_mask = []
    counter = 0
    for k, m in enumerate(model.modules()):
       
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.shape[0]
            weight_copy = m.weight.data.abs().mean((1,2,3)).clone()
            if criterion == 'weight':
                mask = weight_copy.gt(thre.cuda()).float().cuda()
            elif criterion == 'l1':
                mask = feature_maps[counter].abs().mean((0,2,3)).gt(thre.cuda()).float().cuda()
                counter += 1
            elif criterion == 'l2':
                mask = (feature_maps[counter].abs() ** 2).mean((0,2,3)).gt(thre.cuda()).float().cuda()
                counter += 1
            elif criterion == 'saliency':
                mask = (feature_maps[counter] * torch.autograd.grad(loss, feature_maps[counter], only_inputs=True, retain_graph=True)[0].detach()).abs().mean((0,2,3)).gt(thre.cuda()).float().cuda()
                counter += 1
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            #m.weight.data.mul_(mask.view(-1, 1,1,1).repeat(1, *m.weight.data.shape[1:]))
            #m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    pruned_ratio = pruned / total
    p_flops += 2 * total #(minus and sum)

    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    print('Pre-processing Successful!')

    print("Cfg:")
    print(cfg)


    newmodel = resnet18(num_classes=10, cfg=cfg)


    model.load_state_dict(init_weight, strict=False)

    newmodel.to(next(model.parameters()).device)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())

    useful_i = []
    for i, module in enumerate(old_modules):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
            useful_i.append(i)
    temp = []
    for i, item in enumerate(useful_i):
        temp.append(old_modules[item])

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0

    cfg_mask = cfg_mask[0:7] + cfg_mask[8:12] + cfg_mask[13:17] + cfg_mask[18:20]
    downsample = [8, 13, 18]
    last_block = [3, 5, 7, 10, 12, 15, 17, 20]
    for layer_id in range(len(temp)):
        m0 = old_modules[useful_i[layer_id]]
        m1 = new_modules[useful_i[layer_id]]
        # print(m0)
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(layer_id_in_cfg, len(cfg_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[useful_i[layer_id+2]] # channel selection
                assert isinstance(m2, channel_selection)
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif bn_count in downsample:
                # If the current layer is the downsample layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                
            elif bn_count in last_block:
                # If the current layer is the last conv-bn layer in block, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                if bn_count + 1 in downsample:
                    m2 = new_modules[useful_i[layer_id+3]]
                    assert isinstance(m2, channel_selection)
                else:
                    m2 = new_modules[useful_i[layer_id+1]]
                    assert isinstance(m2, channel_selection) or isinstance(m2, nn.Linear)
                if isinstance(m2, channel_selection):
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in downsample: # downsample
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in last_block:
                # the last convolution in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # end_mask = cfg_mask[-1]
            # idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx0)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    model = newmodel
    return model, model.state_dict()



def prune_model_fillback_time(model, masks, num_classes=10):
    total = 0

    for name, m in model.named_modules():
        if not 'downsample' in name and isinstance(m, nn.Conv2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    counter = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and not 'downsample' in name:
            size = m.weight.data.shape[0]
            if name.startswith('conv1'): # conv1
                bn[index:(index+size)] = torch.ones(size)
            else:   
                bn[index:(index+size)] = masks[name].view(size, -1)[:,0]
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    pruned = 0
    cfg = []
    cfg_mask = []
    counter = 0
    for k, (name, m) in enumerate(model.named_modules()):
       
        if isinstance(m, nn.Conv2d) and not 'downsample' in name:
            size = m.weight.data.shape[0]
            if name.startswith('conv1'): # conv1
                mask = torch.ones(size).float().cuda()
            else:
                mask = masks[name].view(size, -1)[:,0].gt(0).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            #m.weight.data.mul_(mask.view(-1, 1,1,1).repeat(1, *m.weight.data.shape[1:]))
            #m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    pruned_ratio = pruned / total
    p_flops += 2 * total #(minus and sum)

    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    print('Pre-processing Successful!')

    print("Cfg:")
    print(cfg)


    newmodel = resnet18(num_classes=num_classes, cfg=cfg)


    #model.load_state_dict(init_weight, strict=False)

    newmodel.to(next(model.parameters()).device)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())

    useful_i = []
    for i, module in enumerate(old_modules):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
            useful_i.append(i)
    temp = []
    for i, item in enumerate(useful_i):
        temp.append(old_modules[item])

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0


    downsample = [8, 13, 18]
    last_block = [3, 5, 7, 10, 12, 15, 17, 20]
    for layer_id in range(len(temp)):
        m0 = old_modules[useful_i[layer_id]]
        m1 = new_modules[useful_i[layer_id]]
        # print(m0)
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(layer_id_in_cfg, len(cfg_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[useful_i[layer_id+2]] # channel selection
                assert isinstance(m2, channel_selection)
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif bn_count in downsample:
                # If the current layer is the downsample layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
            elif bn_count in last_block:
                # If the current layer is the last conv-bn layer in block, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                if bn_count + 1 in downsample:
                    m2 = new_modules[useful_i[layer_id+3]]
                    assert isinstance(m2, channel_selection)
                else:
                    m2 = new_modules[useful_i[layer_id+1]]
                    assert isinstance(m2, channel_selection) or isinstance(m2, nn.Linear)
                if isinstance(m2, channel_selection):
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in downsample: # downsample
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in last_block:
                # the last convolution in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # end_mask = cfg_mask[-1]
            # idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx0)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

    model = newmodel
    return model


def prune_model_fillback_res50_time(model, masks, num_classes=200):
    total = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    counter = 0
    print(masks.keys())
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.shape[0]
            if name.startswith('conv1'): # conv1
                bn[index:(index+size)] = torch.ones(size)
            else:   
                bn[index:(index+size)] = masks[name].view(size, -1)[:,0]
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    p_flops += total * np.log2(total) * 3
    pruned = 0
    cfg = []
    cfg_mask = []
    counter = 0
    for k, (name, m) in enumerate(model.named_modules()):
       
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.shape[0]
            if name.startswith('conv1'): # conv1
                mask = torch.ones(size).float().cuda()
            else:
                mask = masks[name].view(size, -1)[:,0].gt(0).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            #m.weight.data.mul_(mask.view(-1, 1,1,1).repeat(1, *m.weight.data.shape[1:]))
            #m.bias.data.mul_(mask)
            num = int(torch.sum(mask))
            if num != 0:
                cfg.append(num)
                cfg_mask.append(mask.clone())
            elif num == 0:
                cfg.append(1)
                _mask = mask.clone()
                _mask[0] = 1
                cfg_mask.append(_mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            # cfg.append('M')
            pass

    pruned_ratio = pruned / total
    p_flops += 2 * total #(minus and sum)

    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    print('Pre-processing Successful!')

    print("Cfg:")
    print(cfg)

    original_cfg_mask = cfg_mask


    downsample = [5, 15, 28, 47] #[11, 37, 71, 121]
    last_block = [4, 8, 11, 14, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 50, 53]
    new_cfg = []
    for i, item in enumerate(cfg):
        if (i+1) in downsample:
            new_cfg[-1] = max(new_cfg[-1], item)
        # elif i in last_block:
        #     new_cfg[-1] = min(new_cfg[-1], item)
        else:
            new_cfg.append(item)
    new_cfg = [cfg[0]] + new_cfg
    print(new_cfg)
    from models.resnet50_cfg import resnet50_official
    newmodel = resnet50_official(pretrained=False, cfg=new_cfg, imagenet=True, num_classes=num_classes)
    newmodel.cuda()

    return newmodel