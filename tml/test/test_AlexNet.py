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
   from tml.module.AlexNet import *

   net = AlexNet84(128, num_output_layers=1)
   torchinfo.summary(net, input_size=(1, *net.input_shape), device="cpu")

   net = AlexNet28(128, num_output_layers=1)
   torchinfo.summary(net, input_size=(1, *net.input_shape), device="cpu")