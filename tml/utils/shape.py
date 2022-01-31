#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-01-2022 17:07:34

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

def as_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)
