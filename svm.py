#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:51:49 2018

@author: lukaskemmer
"""

import numpy as np
from scipy.optimize import *

##########################################################
# Load data
##########################################################
data = np.load("Wine-Data.npz")
X_training = data['X_training']
X_test = data['X_test']
y_training = data['y_training']
y_test = data['y_test']

##########################################################
# Optimize
##########################################################

# set parameters
l = 1

# 

res_l_bfgs_b = fmin_l_bfgs_b()
