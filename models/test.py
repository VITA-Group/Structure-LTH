from conv import SparseConv2D
import torch

myconv = SparseConv2D(64, 64, kernel_size=3, padding=1)

weight = torch.ones(64, 64, 3, 3)

print(weight.shape)

myconv.load(weight, None)

myconv(torch.ones(127, 127, 64, 64).cuda())


