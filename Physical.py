import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import reduce
from torch.autograd import Variable
import os
from backbone import Backbone

class TransmissionEstimator(nn.Module):
    def __init__(self, width=15,):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0/255
        self.omega=0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width,stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)

    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width//2, self.width//2,self.width//2, self.width//2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self,I,dc):
        n,c,h,w = I.shape
        flat_I = I.view(n, c, -1)
        flat_dc = dc.view(n, 1, -1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:, :, :int(h*w*self.p)]
        searchidx = searchidx.expand(-1, 3, -1)
        searched = torch.gather(flat_I, dim=2, index=searchidx)
        return torch.max(searched, dim=2, keepdim=True)[0].unsqueeze(3)

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        return A


class DehazingNet(nn.Module):
    def __init__(self, min_beta=0.04, max_beta=0.2, min_d=0.3, max_d=5, use_dc_A=True):
        super(DehazingNet, self).__init__()
        self.transmission_estimator = TransmissionEstimator()

        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta
        self.MIN_D = min_d
        self.MAX_D = max_d
        self.use_dc_A = True if use_dc_A == 1 else False
        self.backbone = Backbone()

    def forward_get_A(self, x): # output A: N,3,1,1
        if self.use_dc_A:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        else:
            A = x.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

        return A

    def forward(self, x_0):

        t = self.backbone(x_0)

        t = ((torch.tanh(t) + 1) / 2)
        t = t.clamp(0.05,0.95)

        A = self.forward_get_A(x_0)

        return ((x_0-A)/t + A).clamp(0, 1)


if __name__ == '__main__':
    n = torch.randn((1, 3, 256, 256)).to('cuda')
    net = DehazingNet().to('cuda')
    from thop import profile
    from thop import clever_format

    flops, params = profile(net, inputs=(n,))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
