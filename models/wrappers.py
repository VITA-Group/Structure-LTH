import time
import torch.nn as nn
class TimeWrapper(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
    
    def forward(self, x):
        start = time.time()
        output = self.conv(x)
        end = time.time()
        print(end - start)
        return output

class TimeWrappedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        start = time.time()
        output = super().forward(x)
        end = time.time()
        print(end - start)
        return output