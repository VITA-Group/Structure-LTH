import os
import torch
import matplotlib.pyplot as plt
checkpoint_4 = torch.load(os.path.expanduser("~/data/new_mask_4.pth.tar"))
checkpoint_10 = torch.load(os.path.expanduser("~/data/new_mask_10.pth.tar"))
os.makedirs("vis", exist_ok=True)

def process_heatmap_data(data):
    masks = []
    for i, mask_name in enumerate(data):
        mask = data[mask_name]
        if i >= 10:
            masks.append(mask.view(mask.shape[0], -1))

    masks = torch.cat(masks, 0)
    print(masks.shape)
    return masks

map_imp = process_heatmap_data(checkpoint_10['imp'])
map_refill = process_heatmap_data(checkpoint_4['refill'])
map_regroup = process_heatmap_data(checkpoint_10['regroup'])



for i in range(3):
    plt.figure(figsize=(18,2))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.imshow(map_imp[i*512:(i+1)*512,:], cmap='viridis_r')
    plt.xticks([],[])
    plt.yticks([],[])
    #plt.ylabel(y_torch, fontsize=30)
    plt.savefig(f'vis/IMP_heatmap_{i}.svg', bbox_inches='tight')
    plt.close()


for i in range(3):
    plt.figure(figsize=(18,2))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.imshow(map_refill[i*512:(i+1)*512,:], cmap='viridis_r')
    #plt.xticks([22,22+56,22+56+160,22+56+160+176],[])
    plt.xticks([],[])
    plt.yticks([],[])
    #plt.ylabel(y_zico, fontsize=30)
    plt.savefig(f'vis/refill_heatmap_{i}.svg', bbox_inches='tight')
    plt.close()


for i in range(3):
    plt.figure(figsize=(18,2))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.imshow(map_regroup[i*512:(i+1)*512,:], cmap='viridis_r')
    #plt.xticks([22,22+56,22+56+160,22+56+160+176],[])
    plt.xticks([],[])
    plt.yticks([],[])
    #plt.ylabel(y_nips, fontsize=30)
    plt.savefig(f'vis/regroup_heatmap_{i}.svg', bbox_inches='tight')
    plt.close()