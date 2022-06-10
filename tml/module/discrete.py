#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import torch
import torch.nn as nn
from functools import reduce

from ..utils.shape import as_shape
from .join import JoinProduct


__all__ = ('OneHotEmbedding', 'DiscreteLinear', 'DiscreteOneHot')

class OneHotEmbedding(nn.Module):
    """ 
        Simple one hot embedding for integer values.
        
        Example: 
        '''
            embed = OneHotEmedding(3)
            x = embed(torch.Tensor([0]))
            >> Tensor([[1,0,0]])
        '''
    """
    def __init__(self, size):
        super().__init__()
        self.register_buffer("eye", torch.eye(size, size))
    
    def forward(self, index):
        return self.eye[index]

class DiscreteOneHot(nn.Module):
    """
        Create an action embedding of each action 'a' by 'v = linear(onehot(a))'. The resulting action embedding 
        vector 'v' is then joined with the input 'x' using 'join' which defaults to (elementwise) prod(x, v).
    """

    def __init__(self, input_shape, action_shape, output_shape=None, bias=True, join=None):
        super().__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape) if output_shape is not None else self.input_shape
        self.action_shape = as_shape(action_shape)
        
        self.action_linear = nn.Linear(self.action_shape[0], self.output_shape[0], bias=bias)
        self.onehot_embedding = OneHotEmbedding(self.action_shape[0])
        self.join = join if join is not None else JoinProduct()

    def forward(self, x, a):
        assert reduce(lambda x,y: x*y, a.shape) == a.shape[0] # action should only have singleton dims after dim 0
        a = a.view(a.shape[0])
        a = self.onehot_embedding(a)
        a = self.action_linear(a)
        return self.join([x, a])

class DiscreteLinear(nn.Module):
    """
        Linearly transform input based on a discrete action value. Each possible
        action 'a' [0-N] has learned transformation of the input 'x'.
    """

    def __init__(self, input_shape, action_shape, output_shape=None, bias=True):
        super(DiscreteLinear, self).__init__()
        self.input_shape = as_shape(input_shape)
        self.output_shape = as_shape(output_shape) if output_shape is not None else self.input_shape
        self.action_shape = as_shape(action_shape)

        self.weight = torch.nn.parameter.Parameter(torch.Tensor(self.action_shape[0], self.output_shape[0], self.input_shape[0]))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(self.action_shape[0], self.output_shape[0])) if bias else None
        
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x, a):
        assert x.shape[0] == x.shape[0]
        a = a.long() # ensure `a` can be used as an index
        a = a.view(x.shape[0], 1) # if error, dont use one-hot
        z = (self.weight[a].squeeze(1) @ x.unsqueeze(-1)).squeeze(-1)        
        if self.bias is not None:
            z += self.bias[a].squeeze(1)
        return z