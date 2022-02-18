#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 02-02-2022 15:03:54

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from numpy import identity
import torch
import torch.nn as nn

from torch.nn import Module

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

__all__ = ()

# pytorch version check
assert hasattr(nn.Module(), "_modules")

# add extension methods to pytorch's nn.Module, namely insert, update an delete

def replace_module(self, name: str, module: Optional[Module]) -> None:
    r"""Replaces a child module. 

    The module can be accessed as an attribute using the given name.

    Warning: replacing a child module with an imcompatible module will break its parent!.

    Args:
        name (string): name of the child module. The child module can be
            accessed from this module using the given name
        module (Module): child module to be added to the module.
    """
    if not isinstance(module, Module) and module is not None:
        raise TypeError("{} is not a Module subclass".format(
            torch.typename(module)))
    elif not isinstance(name, torch._six.string_classes):
        raise TypeError("module name should be a string. Got {}".format(
            torch.typename(name)))
    elif '.' in name:
        raise KeyError("module name can't contain \".\", got: {}".format(name))
    elif name == '':
        raise KeyError("module name can't be empty string \"\"")
    self._modules[name] = module

def delete_module(self, name: str) -> None:
    r"""Delete a child module.

    The module can be accessed as an attribute using the given name.

    Warning: Deleting a child module may break its parent!. This is intended for modules that do not change their inputs dimensionality, for example activation functions, dropout or normalisation modules.
 
    Args:
        name (string): name of the child module. The child module can be
            accessed from this module using the given name
        module (Module): child module to be added to the module.
    """
    if not isinstance(name, torch._six.string_classes):
        raise TypeError("module name should be a string. Got {}".format(
            torch.typename(name)))
    elif not hasattr(self, name) or not name in self._modules:
        raise KeyError("attribute '{}' doesn't exists".format(name))
    elif '.' in name:
        raise KeyError("module name can't contain \".\", got: {}".format(name))
    elif name == '':
        raise KeyError("module name can't be empty string \"\"")
    # replace the module with the identity module, rather than deleting it. A user implemneted forward will not be broken!
    self._modules[name] = nn.Identity() 

nn.Module.replace_module = replace_module
nn.Module.delete_module = delete_module

# insert module can happen in nn.Sequential where modules are called in order. More complex examples might exist TODO? 

def insert_module(self, name: int, module: Optional[Module]) -> None:
    r"""Insert a child module. 

    The module can be accessed as an attribute using the given name.

    Warning: replacing a child module with an imcompatible module will break its parent!.

    Args:
        name (int): name of the child module. The child module can be
            accessed from this module using the given name
        module (Module): child module to be added to the module.
    """
    name = str(name) 
    if not isinstance(module, Module) and module is not None:
        raise TypeError("{} is not a Module subclass".format(
            torch.typename(module)))
    elif not isinstance(name, torch._six.string_classes):
        raise TypeError("module name should be an. Got {}".format(
            torch.typename(name)))
    #elif hasattr(self, name) and name not in self._modules:
    #    raise KeyError("attribute '{}' already exists".format(name))
    elif '.' in name:
        raise KeyError("module name can't contain \".\", got: {}".format(name))
    elif name == '':
        raise KeyError("module name can't be empty string \"\"")
    name = int(name)
    # shift all modules up by one
    modules = [(str(int(k) + 1),v) for k,v in self._modules.items() if int(k) >= name]
    for k,v in modules:
        self._modules[k] = v
    self._modules[str(name)] = module

nn.Sequential.insert_module = insert_module

def delete_module(self, name: int) -> None:
    r"""Delete a child module.

    The module can be accessed as an attribute using the given name.

    Warning: Deleting a child module may break its parent!. This is intended for modules that do not change their inputs dimensionality, for example activation functions, dropout or normalisation modules.
 
    Args:
        name (string): name of the child module. The child module can be
            accessed from this module using the given name
        module (Module): child module to be added to the module.
    """
    name = str(name)
    if not isinstance(name, torch._six.string_classes):
        raise TypeError("module name should be a string. Got {}".format(
            torch.typename(name)))
    elif not hasattr(self, name) or not name in self._modules:
        raise KeyError("attribute '{}' doesn't exists".format(name))
    elif '.' in name:
        raise KeyError("module name can't contain \".\", got: {}".format(name))
    elif name == '':
        raise KeyError("module name can't be empty string \"\"")
    # delete the given module by overriding it with the next in the sequence.
    name = int(name)
    modules = [(str(int(k) - 1),v) for k,v in self._modules.items() if int(k) > name]
    print(self._modules)
    for k,v in modules:
        self._modules[k] = v
    print(self._modules)
    del self._modules[str(len(self._modules)-1)]

nn.Sequential.delete_module = delete_module

"""
def shape_decorator(fun):
    # register shape forward hook
    def shape_hook(module, input, output):
        pass 
        name = module.__name__
        
        #print(module, [i.shape for i in input], output.shape)

    def __init__(self, *args, **kwargs):
        fun(self, *args, *kwargs)
        self._shape_hooks = dict()
        for name, module in self.named_modules():
            self._shape_hooks[name] = module.register_forward_hook(shape_hook)

    return __init__

nn.Module.__init__ = shape_decorator(nn.Module.__init__)
"""



