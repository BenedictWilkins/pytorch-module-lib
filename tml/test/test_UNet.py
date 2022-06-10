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
   from tml.module.UNet import UNet

   net = UNet(3, 2)
   torchinfo.summary(net, input_size=(1, 3, 64, 64), device="cpu")

   net = UNet(3, 2, exp=4, batch_normalize=False)
   torchinfo.summary(net, input_size=(1, 3, 64, 64),  device="cpu", depth=5)

  

    
