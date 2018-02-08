#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:04:57 2018

@author: andrea
"""
import numpy as np
import matplotlib.pyplot as plt

def error_shade(Y, x=None, axis=0, error='std', alpha=0.5):
    if x is None:
        x = range(np.shape(Y, axis=1-axis))
    mean = Y.mean(axis=axis)
    error = Y.std(axis=axis)
    plt.plot(x, mean)
    plt.fill_between(x, mean-error, mean+error, alpha=alpha)