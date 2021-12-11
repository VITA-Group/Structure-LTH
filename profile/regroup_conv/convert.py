import torch
import sys
import os
a = torch.load(sys.argv[1], map_location="cpu")
state = a['state_dict']

print(state.keys())
masks = {}
for key in state:
    if 'conv' in key:
        k = state[key]
        masks[key + "_mask"] = (k.abs() != 0).float()

torch.save(masks, sys.argv[1] + "_converted")
