#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 19-05-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn

from .view import Flatten
from ..utils import as_shape


__all__ = ("DiagLinear",)

class DiagLinear(nn.Module):

    def __init__(self, input_shape, action_shape):
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.action_shape = as_shape(action_shape)
        assert len(self.action_shape) == 1

        if len(input_shape) > 1:
            self.flatten_in = Flatten(input_shape)
            self.action_linear = nn.Linear(self.action_shape[0], self.flatten_in.output_shape[0])
            self.flatten_out = self.flatten_in.inverse()
            self.forward = self._forward_flatten
        else:
            self.action_linear = nn.Linear(self.action_shape[0], self.input_shape[0])
            self.forward = self._forward    

    def _forward(self, x, a):
        return x * self.action_linear(a)

    def _forward_flatten(self, x, a):
        x = self.flatten_in(x)
        y = self._forward(x, a)
        y = self.flatten_out(y)
        return y