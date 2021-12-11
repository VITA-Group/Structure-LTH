import torch
import sys

a = torch.load(sys.argv[1])['model']


for k in a.keys():
  if len(a[k].shape) == 4 and a.find('weight') >= 0:
    print(k)
    b = a[k]
    OC, IC, KH, KW = b.shape
    cc = 0
    for i in range(OC):
      for j in range(IC):
        for k in range(KH):
          for m in range(KW):
            if b[i,j,k,m] != 0:
              cc += 1

    ratio = OC * IC * KH * KW / cc
    print(ratio)
