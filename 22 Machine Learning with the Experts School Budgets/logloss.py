#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:07:10 2019

@author: amin
"""

import numpy as np

def compute_log_loss(predicted, actual, eps=1e-14):
     """ Computes the logarithmic loss between predicted and
     actual when these are 1D arrays.
     
     :param predicted: The predicted probabilities as floats between 0-1
     :param actual: The actual binary labels. Either 0 or 1.
     :param eps (optional): log(0) is inf, so we need to offset our
     predicted values slightly by eps from 0 or 1.
     """
     
     predicted = np.clip(predicted, eps, 1 - eps)
     # we use the clip function which sets a maximum and minimum value for elements 
     #in an array. Since log of 0 is negative infinity we want to offset our predictions so slightly
     # from being exactly 1 or exactly 0 so that the score remains a real number
     loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
     
     return loss