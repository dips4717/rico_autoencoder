#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:16:06 2020

@author: dipu
"""


from .model_strided_emb512 import CAE_strided_dim512
from .model_strided import CAE_strided
from .model_upsample import CAE_upsample
from .model_upsample_emb512 import CAE_upsample_dim512


__factory = {
        'strided': CAE_strided,
        'strided_512': CAE_strided_dim512,
        'upsample': CAE_upsample,
        'upsample_512': CAE_upsample_dim512}


def create(name):
    if name not in __factory:
        raise KeyError ("Unknown network: ", name )
    return __factory[name]()