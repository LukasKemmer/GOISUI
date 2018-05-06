#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 23:22:51 2018

@author: lukaskemmer
"""

import numpy as np
import matplotlib.pyplot as plt

##########################################################
# Load data
##########################################################
data = np.load("Approx-Data.npz")
x = data['x']
y = data['y']

##########################################################
# b) Bestimme Minimalprunkt für P1 und P2
##########################################################




##########################################################
# c) Plotte Datenpunkte sowie f1, f2
##########################################################

plt.style.use('seaborn-ticks')
fig, ax = plt.subplots()
ax.plot(x, y, 'o')
plt.title("Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


