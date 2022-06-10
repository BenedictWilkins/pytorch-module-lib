#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 08-04-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from typing import Callable, Tuple, List
import torch
import torch.nn as nn
from ..utils.shape import as_shape
from .view import Flatten, View
from .MLP import MLP

class AlexNet(nn.Sequential):

    def __init__(self, input_shape : Tuple[int], conv_layers : List[nn.Module], mlp_layers : List[nn.Module], flatten : bool = True):
        super().__init__()
        self.input_shape = as_shape(input_shape)
        with torch.no_grad():
            conv_output_shape = as_shape(nn.Sequential(*conv_layers)(torch.empty((2, *self.input_shape))).shape[1:])
        if len(conv_output_shape) > 1 and flatten:
            mlp_layers = [Flatten(conv_output_shape), *mlp_layers]
        super().__init__(*conv_layers, *mlp_layers)
        with torch.no_grad():
            self.output_shape =  as_shape(self.forward(torch.empty((2, *self.input_shape))).shape[1:])

class AlexNet28(AlexNet):

    def __init__(self, output_shape : Tuple[int], 
                    input_shape : Tuple[int] = (3,28,28), 
                    hidden_shape : Tuple[int] = (512,), 
                    num_output_layers : int = 2, 
                    dropout : float = 0.5, 
                    output_activation : Callable = None):
        output_shape = as_shape(output_shape) 
        input_shape = as_shape(input_shape)
        assert input_shape[1:] == (28,28)
        conv_block = [
            nn.Conv2d(input_shape[0], 16, 7, 1), nn.LeakyReLU(), 
            nn.Conv2d(16, 32, 7, 2), nn.LeakyReLU(), 
            nn.Conv2d(32, 64, 3, 1), nn.LeakyReLU(), 
            nn.Conv2d(64, 64, 3, 1), nn.LeakyReLU(),
        ]
        mlp_block = []
        if num_output_layers > 0:
            mlp_block = [m for m in MLP(1024, hidden_shape, output_shape, dropout=dropout, num_layers=num_output_layers, output_activation=output_activation)]
        super().__init__(input_shape, conv_block, mlp_block)

class AlexNet84(AlexNet):
    
    def __init__(self, output_shape : Tuple[int], 
                    input_shape : Tuple[int] = (3,84,84), 
                    hidden_shape : Tuple[int] = (512,), 
                    num_output_layers : int = 2, 
                    dropout : float = 0.5, 
                    output_activation : Callable = None):
        output_shape = as_shape(output_shape) 
        input_shape = as_shape(input_shape)
        assert input_shape[1:] == (84,84)
        conv_block = [
            nn.Conv2d(input_shape[0],16,kernel_size=6,stride=1), nn.LeakyReLU(),
            nn.Conv2d(16,32,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(32,64,kernel_size=6,stride=1), nn.LeakyReLU(),
            nn.Conv2d(64,128,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=2), nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size=5,stride=1), nn.LeakyReLU(),
        ]
        mlp_block = []
        if num_output_layers > 0:
            mlp_block = [m for m in MLP(128 * 2 * 2, hidden_shape, output_shape, dropout=dropout, num_layers=num_output_layers, output_activation=output_activation)]
            
        super().__init__(input_shape, conv_block, mlp_block)

