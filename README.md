# Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper: [Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets](http://arxiv.org/abs/2202.04736)

Tianlong Chen, Xuxi Chen, Xiaolong Ma, Yanzhi Wang, Zhangyang Wang

## Overview

In this paper, we demonstrate the first positive result that a structurally sparse winning ticket can be effectively found in general. The core idea is to append “post-processing techniques” after each round of (unstructured) IMP, to enforce the formation of structural sparsity.

Specifically, we first “re-fill” pruned elements back in some channels deemed to be important, and then “re-group” non-zero elements to create flexible group-wise structural patterns. Both our identified channel- and group-wise structural subnetworks win the lottery, with substantial inference speedups readily supported by practical hardware.

Extensive experiments, conducted on diverse datasets across multiple network backbones, consistently validate our proposal, showing that the hardware acceleration roadblock of LTH is now removed. Detailed results are referred to our [paper](http://arxiv.org/abs/2202.04736).

## Method

Overview of our proposals including refilling, refilling+, and regrouping, which turn unstructured sparse mask into channel-wise and group-wise structured sparse masks.

![Methods](Figs/Methods.png)

## Prerequisites

Our code works with general version of PyTorch. We suggest use versions that are compatible with CUDA 10.2 since the profiling code requires CUDA 10.2.

For example:

```bash
conda create -n structlth python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
conda install matplotlib
pip install advertorch tqdm networkx
```

or

```bash
conda env create -f environment.yml
```

Please notice that we need `nvcc` to be installed.

### Data Preparation

Most of the datasets will be downloaded automatically. To download the Tiny-ImageNet, please refer to [this link](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4).

## Experiments

### Checkpoints

The relevant files / checkpoints can be found in [this folder](https://www.dropbox.com/sh/0j9p3hfbmm9wn3r/AABi3hI_2esiw40JsKaX1teka?dl=0) and [this folder] (https://drive.google.com/drive/folders/19PSxCZ_q0eNmdO4AZYqZSNnJPrTErxqg?usp=sharing).

### Finding Lottery Tickets with IMP (ResNet-18)

```bash
python -u main_imp.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.2_s1_rewind_16 --init pretrained_model/res18_cifar10_1_init.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --epoch 160 --decreasing_lr 80,120 --rewind_epoch 16 --weight_decay 1e-4 --batch_size 128
```

### Retrain Networks with Refill (ResNet-18)

```bash
i=1 # Take 1 as an example
python -u main_eval_fillback.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir output --pretrained resnet18_cifar10_lt_0.2_s1_rewind_16/1checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.2_s1_rewind_16/${i}checkpoint.pth.tar --fc --prune-type lt --seed 1 --epoch 160 --decreasing_lr 80,120 --weight_decay 1e-4 --batch_size 128 --lr 0.1 
```

### Retrain Networks with Regroup (ResNet-18)

```bash
i=1
python -u main_eval_regroup.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir output --pretrained resnet18_cifar10_lt_0.2_s1_rewind_16/1checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.2_s1_rewind_16/${i}checkpoint.pth.tar --fc --prune-type lt --seed 1 --epoch 160 --decreasing_lr 80,120 --weight_decay 1e-4 --batch_size 128 --lr 0.1 
```

## Profiling

The code for profiling is under `profile`.

To calculate the time of regroup conv, `cd profile/regroup_conv` and `python split.py <checkpoint> <dir_to_save>`. For each extracted sparse mask, run `python conv.py --kernel_file <sparse_mask_checkpoint>`.

To calculate the time of cudnn conv, `cd profile/cudnn_conv` and run `python conv.py --kernel_file <sparse_mask_checkpoint>`.

### Example of end-to-end inference

Ours (regroup): 
```bash
CUDA_VISIBLE_DEVICES=0 python -u end-to-end.py --data datasets/cifar10 --dataset cifar10 --seed 1 --arch vgg16_bn --epoch 160 --lr 0.1 --decreasing_lr 80,120 --save_dir vgg16_bn_cifar10_lt_0.2_rewind_regroup_vgg_time --checkpoint vgg16_bn_cifar10_lt_0.2_rewind_imp/10checkpoint.pth.tar --prune-type lt --reuse
```
Original
```bash
CUDA_VISIBLE_DEVICES=0 python -u end-to-end.py --data datasets/cifar10 --dataset cifar10 --seed 1 --arch vgg16_bn --epoch 160 --lr 0.1 --decreasing_lr 80,120 --save_dir vgg16_bn_cifar10_lt_0.2_rewind_regroup_vgg_time --checkpoint vgg16_bn_cifar10_lt_0.2_rewind_imp/10checkpoint.pth.tar --prune-type lt --use-original
```

## Todo

- [] Upgrade codes to support CUDA 11.x.
- [] Update commands for other experiments.

## Aknowledgement

Many thanks Prof. Jiang from [paper](https://doi.org/10.1145/3410463.3414648) for providing implementations of acceleration and helpful discussions!

## Citation

```latex
@misc{chen2022coarsening,
      title={Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets}, 
      author={Tianlong Chen and Xuxi Chen and Xiaolong Ma and Yanzhi Wang and Zhangyang Wang},
      year={2022},
      eprint={2202.04736},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

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
