#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import torch.nn as nn
from ..utils.shape import as_shape
from .view import Flatten

class MLP(nn.Sequential):

    def __init__(self, input_shape, hidden_shape, output_shape, layers=2, hidden_activation=nn.LeakyReLU, output_activation=None):
        self.input_shape = as_shape(input_shape)
        self.hidden_shape = as_shape(hidden_shape)
        self.output_shape = as_shape(output_shape)
        assert layers >= 2 
        assert hidden_activation is not None
         
        modules = []
        if len(self.input_shape) > 1:
            modules.append(Flatten(self.input_shape))
        shapes = [np.prod(self.input_shape), *([self.hidden_shape[0]] * (layers - 1)), self.output_shape[0]]
        for s1, s2 in zip(shapes[:-1], shapes[1:]):
            modules.append(nn.Linear(s1, s2))
            modules.append(hidden_activation())
        modules = modules[:-1]
        if output_activation is not None:
            modules.append(output_activation())

        super().__init__(*modules)