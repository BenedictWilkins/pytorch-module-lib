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
    from tml.module.MLP import *

    net = MLP(16,8,4, dropout=0.5,output_activation=nn.Sigmoid)
    torchinfo.summary(net, input_size=(1, *net.input_shape), device="cpu")