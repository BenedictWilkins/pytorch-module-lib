#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 

    Modules to join data

   Created on 18-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn
from functools import reduce
from typing import List
from ..utils.shape import as_shape

__all__ = ("JoinConcatenate", "JoinSum", "JoinProduct", "JoinAttention")

class JoinConcatenate(nn.Module):

    def forward(self, x  : List[torch.Tensor]): # [batch_size, *input_shape]
        return torch.cat(x, dim=1) # concatenate over first dim of input_shape

class JoinSum(nn.Module):

    def forward(self, x : List[torch.Tensor]):
        return sum(x)

class JoinProduct(nn.Module):

    def forward(self, x : List[torch.Tensor]):
        return reduce(lambda x,y : x * y, x)

class JoinAttention(nn.Module):

    def __init__(self, input_shape, num_heads=1, residual = True, q_index=0):
        super().__init__()
        self.input_shape = as_shape(input_shape)
        assert len(self.input_shape) == 1 # flatten input...
        self.attention = nn.MultiheadAttention(self.input_shape[0], num_heads=num_heads, batch_first=False)
        self.layer_norm = nn.LayerNorm(self.input_shape[0]) if residual else None
        self.residual = residual
        self.q_index = q_index

    def forward(self, x : List[torch.Tensor]):
        """
            Uses pytorch MultiHeadAttention to join input, the input 'x' is treated as the source sequence and forms the keys and values. 
            The query is the feature vector specified by 'q_index'. 
        Args:
            x (List[torch.Tensor]): 
        """
        x = torch.stack(x, dim=0) # [window_size, batch_size, *input_shape]
        k, v, q = x, x, x[self.q_index].unsqueeze(0)
        y, _ = self.attention.forward(q, k, v, need_weights=False) # [1, batch_size, *input_shape]
        y = self.layer_norm(q + y) if self.residual else y
        return y.squeeze(0)


    