#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 08-04-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

if __name__ == "__main__":

   import torchinfo
   from tml.module.AE import *

   net = AEConv28((1,28,28), num_linear_layers=1)
   torchinfo.summary(net, input_size=(1, *net.input_shape), device="cpu")