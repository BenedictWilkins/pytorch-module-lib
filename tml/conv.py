#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch.nn as nn

__all__ = ("ResBlock2D", )

class ResBlock2D(nn.Module):

    def __init__(self, in_channel, channel):
        super(ResBlock2D, self).__init__()
        self.in_channel = in_channel
        self.channel = channel
        self.c1 = nn.Conv2d(in_channel, channel, 3, 1, 1)
        self.r1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(channel, in_channel, 1)
        
    def forward(self, x):
        x_ = self.c2(self.r1(self.c1(x)))
        x_ += x
        return x_