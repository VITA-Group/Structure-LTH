# Lottery Tickets can have Structural Sparsity
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper: [Lottery Tickets can have Structural Sparsity](https://openreview.net/pdf?id=oZe7Zdia1H5)

Tianlong Chen, Xuxi Chen, Xiaolong Ma, Yanzhi Wang, Zhangyang Wang

## Overview

In this paper, we demonstrate the first positive result that a structurally sparse winning ticket can be effectively found in general. The core idea is to append “post-processing techniques” after each round of (unstructured) IMP, to enforce the formation of structural sparsity. 

Specifically, we first “re-fill” pruned elements back in some channels deemed to be important, and then “re-group” non-zero elements to create flexible group-wise structural patterns. Both our identified channel- and group-wise structural subnetworks win the lottery, with substantial inference speedups readily supported by practical hardware. 

Extensive experiments, conducted on diverse datasets across multiple network backbones, consistently validate our proposal, showing that the hardware acceleration roadblock of LTH is now removed. Detailed results are referred to our [paper](https://openreview.net/pdf?id=oZe7Zdia1H5). 



## Method

Overview of our proposals including refilling, refilling+, and regrouping, which turn unstructured sparse mask into channel-wise and group-wise structured sparse masks.

![](Figs/Methods.png)



## Prerequisites



## Experiments

### Finding Lottery Tickets with IMP

```bash
python -u main_imp.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2_s1_rewind_16 --init pretrained_model/res18_cifar10_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --epoch 160 --decreasing_lr 80,120 --rewind_epoch 16 --weight_decay 1e-4 --batch_size 128
```

### Retrain Networks with Refill

```bash
i=1
python -u main_eval_fillback.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir  --pretrained resnet18_cifar10_lt_0.2_s1_rewind_16/1checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.2_s1_rewind_16/${i}checkpoint.pth.tar --fc --prune-type lt --seed 1 --epoch 160 --decreasing_lr 80,120 --weight_decay 1e-4 --batch_size 128 --lr 0.1 
```

### Retrain Networks with Regroup

```bash
i=1
python -u main_eval_regroup_retrain.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir  --pretrained resnet18_cifar10_lt_0.2_s1_rewind_16/1checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.2_s1_rewind_16/${i}checkpoint.pth.tar --fc --prune-type lt --seed 1 --epoch 160 --decreasing_lr 80,120 --weight_decay 1e-4 --batch_size 128 --lr 0.1 
```



## Aknowledgement

Many thanks Prof. Jiang from [paper](https://doi.org/10.1145/3410463.3414648) for providing implementations of acceleration and helpful discussions!



## Citation

```
TBD

@inproceedings{Rumi2020acc,
author = {Rumi, Masuma Akter and Ma, Xiaolong and Wang, Yanzhi and Jiang, Peng},
title = {Accelerating Sparse CNN Inference on GPUs with Performance-Aware Weight Pruning},
year = {2020},
isbn = {9781450380751},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3410463.3414648},
doi = {10.1145/3410463.3414648},
booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
pages = {267–278},
numpages = {12},
keywords = {cnn pruning, sparse convolution, gpus},
location = {Virtual Event, GA, USA},
series = {PACT '20}
}
```