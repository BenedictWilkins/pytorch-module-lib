#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 08-04-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

if __name__ == "__main__":

   import torch
   import torchinfo
   from tml.module.hyper import DiagLinear
   input_shape = (4,)
   action_shape = (2,)
   net = DiagLinear(input_shape, action_shape)
   torchinfo.summary(net, input_size=[(1, *input_shape), (1, *action_shape)], device="cpu")
   x = net(torch.empty(1, *input_shape), torch.empty(1, *action_shape))
   assert tuple(x.shape) == (1,*input_shape)


   input_shape = (4,4)
   action_shape = (2,)
   net = DiagLinear(input_shape, action_shape)
   torchinfo.summary(net, input_size=[(1, *input_shape), (1, *action_shape)], device="cpu")
   x = net(torch.empty(1, *input_shape), torch.empty(1, *action_shape))
   assert tuple(x.shape) == (1,*input_shape)