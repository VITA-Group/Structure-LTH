import torch
import sys
import os
a = torch.load(sys.argv[1], map_location="cpu")
try:
    state = a['state_dict']
except:
    state = a

print(state.keys())
masked = False
for key in state:
    if 'mask' in key:
        try:
            k = state[key[:-4] + "orig"] * state[key[:-4] + "mask"]
        except:
            k = state[key[:-4] + "mask"]

        os.makedirs(sys.argv[2], exist_ok=True)
        torch.save(k, sys.argv[2] + "/" + key[:-5] + ".pth.tar", _use_new_zipfile_serialization=False)
        masked = True

if not masked:
    for key in state:
        print(state[key].dim())
        if 'conv' in key or state[key].dim() > 2:
            k = state[key]
            os.makedirs(sys.argv[2], exist_ok=True)
            torch.save(k, sys.argv[2] + "/" + key + ".pth.tar", _use_new_zipfile_serialization=False)
            masked = True