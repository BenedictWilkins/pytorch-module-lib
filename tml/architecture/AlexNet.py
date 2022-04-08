#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 08-04-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn
from ..utils.shape import as_shape
from ..module.view import View
from ..module.MLP import MLP

class AlexNet84(nn.Sequential):
    
    def __init__(self, output_shape, input_shape=(3,84,84), hidden_shape=(512,), num_output_layers=2, dropout=0.5, output_activation=None):
        self.output_shape = as_shape(output_shape)
        self.input_shape = as_shape(input_shape)
        assert self.input_shape[1:] == (84,84)
        conv_block = [
            nn.Conv2d(3,16,kernel_size=6,stride=1), nn.LeakyReLU(),
            nn.Conv2d(16,32,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(32,64,kernel_size=6,stride=1), nn.LeakyReLU(),
            nn.Conv2d(64,128,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=1), nn.LeakyReLU(),
        ]
        mlp_block = [m for m in MLP((128,2,2), hidden_shape, self.output_shape, layers=num_output_layers, output_activation=output_activation)]
        if dropout > 0: # insert dropout
            for i in range(2,len(mlp_block)-1,2):
                mlp_block.insert(i, nn.Dropout(dropout))

        super().__init__(*conv_block, *mlp_block)
