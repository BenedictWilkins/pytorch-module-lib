#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-01-2022 17:03:11

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import inspect

def override_with_instance_variables(func):
    """ 
        Override each of the arguments whose default value is None if it has not been set by the caller
        and is an instance variable. This is just syntatic sugar for writing an if statement inside the method body, e.g.

        def test(self, a=None):
            if a is None:
                a = self.a

        instead use this annotation!

        def test(self, a=None):
            # use a as you like it
        
        instance.test()     # a takes the value of instance.a
        instance.test(a=1)  # a takes the value 1

    Args:
        func (method): method to annotate

    Returns:
        method: annotated method
    """
    spec = inspect.getfullargspec(func)
    args_to_fill = spec.args[-len(spec.defaults):] 
    args_to_fill = [a for a,d in zip(args_to_fill, spec.defaults) if d is None]
    def _owiv(self, *args, **kwargs):
        ukw = {k:getattr(self, k) for k in args_to_fill if k not in kwargs}
        kwargs.update(ukw)
        return func(self, *args, **kwargs)
    return _owiv 