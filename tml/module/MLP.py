#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from typing import Union, Callable, Tuple, List, Dict
from grpc import Call
import numpy as np
import torch.nn as nn
from ..utils.shape import as_shape
from .view import Flatten, View

class MLP(nn.Sequential):

    def __init__(self, 
                    input_shape : Tuple[int], 
                    hidden_shape : Union[Tuple[int], List[Tuple[int]]], 
                    output_shape : Tuple[int], 
                    num_layers : int = 2, 
                    hidden_activation : Callable = nn.LeakyReLU, 
                    output_activation : Callable = None,
                    dropout : Union[float, List[float]] = None):
        """ Create an MLP.

        Args:
            input_shape (Tuple[int]): input shape of the network.
            hidden_shape (Union[Tuple[int], List[Tuple[int]]]): hidden_shape(s), if a integer or tuple (of length 1) this shape will be repeated `num_layers` times, otherwise `num_layers` will be ignored and `hidden_shapes` will be used to build the network.
            output_shape (Tuple[int]): output shape of the network.
            num_layers (int, optional): number of num_layers (nn.Linear + activation) in the network > 1. Defaults to 2.
            hidden_activation (Callable, optional): hidden_activation(s) to use. Defaults to nn.LeakyReLU.
            output_activation (Callable, optional): output activation to use, this is activation used after the final layer. Defaults to None.
            dropout (Union[float, List[float]], optional): dropout prob(s) to use negative, 0 or None values will not apply dropout. A float value > 0 will apply dropout to each layer (except the final layer). Defaults to None.

        Raises:
            ValueError: if any parameters are invalid.
        """
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape)
        
        if isinstance(hidden_shape, int):
            hidden_shape = (hidden_shape,)
        if isinstance(hidden_shape, tuple) and len(hidden_shape) == 1:
            hidden_shape = [as_shape(hidden_shape)] * (num_layers - 1)
        if isinstance(hidden_shape, (list, tuple)):
            num_layers = len(hidden_shape) + 1
            hidden_shape = [as_shape(s) for s in hidden_shape]
            
        assert num_layers > 0
        assert len(hidden_shape) == num_layers - 1

        if isinstance(hidden_activation, type):
            hidden_activation = hidden_activation()
        if isinstance(output_activation, type):
            output_activation = output_activation()

        modules = []
        if len(self.input_shape) > 1:
            modules.append(Flatten(self.input_shape))

        shapes = [as_shape(np.prod(self.input_shape).item()), *hidden_shape]

        if dropout is None:
            dropout = [None] * (num_layers)
        elif isinstance(dropout, (float, int)):
            if dropout > 0:
                dropout = [nn.Dropout(float(dropout))] * (num_layers - 1) + [None]
            else: 
                dropout = [None] * (num_layers)
        elif isinstance(dropout, (list, tuple)):
            dropout = [(nn.Dropout(dp) if dp is not None and dp > 0. else None) for dp in dropout] 
            if len(dropout) < num_layers:
                dropout += ([None] * (num_layers - 1 - len(dropout)))
        else:
            raise ValueError(f"Invalid dropout {dropout}")
        if len(dropout) != num_layers:
            raise ValueError(f"Dropout list {dropout} is too large, should be less or equal to the number of num_layers ({num_layers}) in the MLP.")

        if hidden_activation is None:
            hidden_activation = [None] * (num_layers - 1)
        elif callable(hidden_activation):
            hidden_activation = [hidden_activation] * (num_layers - 1)
        elif isinstance(dropout, (list, tuple)):
            hidden_activation = [(ha() if ha is not None else None) for ha in hidden_activation] 
            if len(hidden_activation) < num_layers - 1:
                hidden_activation += ([None] * (num_layers - 1 - len(hidden_activation)))
        else:
            raise ValueError(f"Invalid hidden_activation {hidden_activation}")
        if len(hidden_activation) != num_layers - 1:
            raise ValueError(f"Hidden activation list {hidden_activation} is too large, should be less than the number of num_layers ({num_layers}) in the MLP.")
        
        for s1, s2, ha, drop in zip(shapes[:-1], shapes[1:], hidden_activation, dropout):
            #print(s1, s2, drop)
            modules.append(nn.Linear(s1[0], s2[0]))
            modules.append(ha)
            modules.append(drop)
        
        modules.append(nn.Linear(shapes[-1][0], np.prod(self.output_shape)))
        if len(self.output_shape) > 1:
            modules.append(View(-1, self.output_shape))
        modules.append(output_activation)
        modules.append(dropout[-1])
        
        modules = [m for m in modules if m is not None]
      
        super().__init__(*modules)