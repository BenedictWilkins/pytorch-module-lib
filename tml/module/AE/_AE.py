#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 10-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from copy import deepcopy
import torch
import torch.nn as nn

from typing import Tuple, Callable

import torchinfo

from ..AlexNet import AlexNet28
from ..view import View

from ...utils import invert

class AEConv28(nn.Module):

    def __init__(self,
                    input_shape : Tuple[int] = (3,28,28), 
                    hidden_shape : Tuple[int] = (512,), 
                    num_linear_layers : int = 2, 
                    dropout : float = 0.5, 
                    share_weights = False,
                    output_activation : Callable = None, 
                    **kwargs):
        super().__init__()
        self.encoder = AlexNet28(input_shape=input_shape, 
                                    output_shape=hidden_shape,
                                    hidden_shape=hidden_shape, 
                                    num_output_layers=num_linear_layers, 
                                    dropout=dropout, 
                                    output_activation=output_activation, 
                                    **kwargs)
        self.input_shape = self.encoder.input_shape
        self.output_shape = self.encoder.output_shape
        self.decoder = invert.Sequential(self.encoder, self.input_shape)
       
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
