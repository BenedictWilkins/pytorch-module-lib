#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 10-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from collections.abc import MutableMapping

__all__ = ("PolyDict",)

class PolyDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._mapping = dict(*args, **kwargs)

    def __getitem__(self, key):
        for cls in key.__mro__:
            if cls in self._mapping:
                return self._mapping[cls]
        raise KeyError(key)

    def __delitem__(self, key):
        del self._mapping[key]

    def __setitem__(self, key, value):
        self._mapping[key] = value

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)
