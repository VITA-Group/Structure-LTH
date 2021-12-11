from models.vgg import vgg16_bn
import torch
import torch.nn as nn
import sys
checkpoint = sys.argv[1] 

from pruning_utils import extract_mask

def prune_model_custom_fillback_time(model, mask_dict, conv1=False, criteria="magnitude", train_loader=None, fillback_rate = 0.0):

    new_mask_dict = {}
    channels = []
    for i, (name,m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                mask = mask_dict[name+'.weight_mask']
                mask = mask.view(mask.shape[0], -1)
                count = torch.sum(mask != 0, 1) # [C]
                #sparsity = torch.sum(mask) / mask.numel()
                num_channel = count.sum().float() / mask.shape[1]
                num_channel = num_channel + (mask.shape[0] - num_channel) * fillback_rate
                print(num_channel)
                int_channel = int(num_channel)
                frac_channel = num_channel - int_channel
                channels.append(int(num_channel) + 1)
                
                if criteria == 'magnitude':
                    mask = mask_dict[name+'.weight_mask']
                    count = m.weight.data.view(mask.shape[0], -1).abs().sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                
                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    mask[torch.where(count == threshold)[0],:int(frac_channel * mask.shape[1])] = 1
                    mask[torch.where(count == threshold)[0],int(frac_channel * mask.shape[1]):] = 0
                            
                #mask = mask.view(*mask_dict[name+'.weight_mask'].shape)
                new_mask_dict[name+'.weight_mask'] = mask
                #prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))

    return new_mask_dict


state_dict = torch.load(checkpoint, map_location="cpu")['state_dict']
model = vgg16_bn(pretrained=False)
current_mask = extract_mask(state_dict)
import copy
current_mask_copy = copy.deepcopy(current_mask)
print(current_mask.keys())
refill_masks = prune_model_custom_fillback_time(model, current_mask_copy)
from pruning_utils import regroup
regroup_masks = {}
current_mask_copy_2 = copy.deepcopy(current_mask)

for key in current_mask_copy_2:
    mask = current_mask_copy_2[key]
    regroup_masks[key] = regroup(mask.view(mask.shape[0], -1))
    print(regroup_masks[key].numel() / (regroup_masks[key].abs() > 0).float().sum())

all_masks = {'refill': refill_masks, 'imp': current_mask, 'regroup': regroup_masks}

torch.save(all_masks, f"{sys.argv[2]}")