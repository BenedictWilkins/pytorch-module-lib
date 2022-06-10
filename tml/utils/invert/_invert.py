#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 11:39:40

    Build inverse PyTorch layers.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from copy import deepcopy
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from .. import shape
from ..polydict import PolyDict

__all__ = (
    "Conv2d", 
    "MaxPool1d", 
    "MaxPool2d", 
    "MaxPool3d", 
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Dropout",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "Sequential",
)

_AMBIGUOUS_SHAPE_MSG = "Ambiguous inverse of layer {0} without shape argument."

def Conv2d(layer, input_shape, share_weights=False, **kwargs):
    convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                                kernel_size=layer.kernel_size, 
                                stride=layer.stride, 
                                padding=layer.padding)

    c_shape = shape.Conv2d(layer, input_shape)
    ct_shape = shape.ConvTranspose2d(convt2d, c_shape)
    dh, dw = input_shape[1] - ct_shape[1], input_shape[2] - ct_shape[2]
    convt2d.output_padding = (dh, dw)
    
    if share_weights:
        convt2d.weight = layer.weight
    return convt2d


def MaxPool1d(layer, **kwargs):
    raise NotImplementedError() #TODO

def MaxPool2d(layer, **kwargs):
    assert isinstance(layer, nn.MaxPool2d)
    return nn.MaxUnpool2d(layer.kernel_size, layer.stride, layer.padding)

def MaxPool3d(layer, **kwargs):
    raise NotImplementedError() #TODO

def Linear(layer, share_weights=False, **kwargs):
    """ Transpose linear layer.

    Args:
        layer (torch.nn.Linear): Linear layer.
        share_weights (bool, optional): should the inverse layer share weights? (bias will not be shared). Defaults to False.

    Returns:
        torch.nn.Linear: inverse layer.
    """
    lt = nn.Linear(layer.out_features, layer.in_features, layer.bias is not None)
    if share_weights:
        lt.weight = nn.Parameter(layer.weight.t())
    return lt

def BatchNorm1d(layer, **kwargs):
    return copy.deepcopy(layer)

def BatchNorm2d(layer, **kwargs):
    return copy.deepcopy(layer)

def BatchNorm3d(layer, **kwargs):
    return copy.deepcopy(layer)

def Dropout(layer, **kwargs):
    return copy.deepcopy(layer)

def AdaptiveAvgPool1d(layer, input_shape, **kwargs):
    raise NotImplementedError()

def AdaptiveAvgPool2d(layer, input_shape, **kwargs):
    assert len(input_shape) == 3 # C,H,W 
    return nn.AdaptiveAvgPool2d(input_shape[1:])

def AdaptiveAvgPool3d(layer, input_shape, **kwargs):
    raise NotImplementedError()

def Sequential(sequential, input_shape, **kwargs):
    shapes = shape.Sequential(sequential, input_shape)

    imodules = []
    shapes = reversed(shapes)
    children = reversed(list(sequential.children()))
    for child, sh in zip(children, shapes):
        imodules.append(invert(child, input_shape=sh, **kwargs)[0])
    return nn.Sequential(*imodules)

def Identity(layer, *args, copy=False, **kwargs):
    return deepcopy(layer) if copy else layer

invert_map = PolyDict({
    #nn.Conv1d: not_implemented,
    nn.Conv2d: Conv2d,
    #nn.Conv3d: not_implemented,
    nn.Linear: Linear,
    nn.Identity: partial(Identity, copy=True),
    nn.Sequential: Sequential,
    nn.LeakyReLU:   partial(Identity, copy=True),
    nn.ReLU:        partial(Identity, copy=True),
    nn.Sigmoid:     partial(Identity, copy=True),
    nn.Tanh:        partial(Identity, copy=True),
    nn.BatchNorm1d: BatchNorm1d,
    nn.BatchNorm2d: BatchNorm2d,
    nn.BatchNorm3d: BatchNorm3d,
    nn.AdaptiveAvgPool1d: AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d: AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d: AdaptiveAvgPool3d,

    nn.MaxPool1d: MaxPool1d,
    nn.MaxPool2d: MaxPool2d,
    nn.MaxPool3d: MaxPool3d,

    nn.Dropout: partial(Identity, copy=True),
})

def invert(*layers, **kwargs):
    """ Inverse a sequence of layers. The returned sequence will be in reverse order, i.e. if the input is l1,l2,l3, the output will be the li3,li2,li1.

    Args:
        share_weights (bool, optional): should the inverse layers share the weights of the original?. Defaults to True.

    Returns:
        list: inverse layers (in reverse order).
    """
    inverse_layers = []
    for layer in reversed(layers):
        if type(layer) in invert_map:
            inverse_layers.append(invert_map[type(layer)](layer, **kwargs))
        elif hasattr(layer, "inverse"):
            inverse_layers.append(layer.inverse(**kwargs))
        else:
            raise ValueError("Failed to get inverse for layer: {0}".format(layer))

    return inverse_layers