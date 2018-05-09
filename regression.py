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
X = np.column_stack((x, np.ones((len(x), 1))))
y = data['y']

##########################################################
# b) Bestimme Minimalprunkt für P1 und P2
##########################################################

# P1 (keine analytische Lösung)


# P2 (beta = (X^T*X)^-1*X^T*Y)
beta_p2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)

##########################################################
# c) Plotte Datenpunkte sowie f1, f2
##########################################################

plt.style.use('seaborn-ticks')
fig, ax = plt.subplots()
ax.plot(X[:,0], y, 'o')
ax.plot(X[:,0], beta_p2[0]*X[:,0]+beta_p2[1], 'k')
plt.title("Linear model ohne Aussreisser")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Data', 'Fit from P1', 'P2'])
plt.show()

##########################################################
# d) Datenpunkte hinzufügen und b), c) wiederholen
##########################################################

# Fuege Ausreisser hinzu
X = np.row_stack((X, np.array([[1, 1], [2, 1]])))
y = np.row_stack((y, np.array([[30], [35]])))

# Bestimme parameter
# P1 (keine analytische Lösung)


# P2 (beta = (X^T*X)^-1*X^T*Y)
beta_p2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)

plt.style.use('seaborn-ticks')
fig, ax = plt.subplots()
ax.plot(X[:,0], y, 'o')
ax.plot(X[:,0], beta_p2[0]*X[:,0]+beta_p2[1], 'k')
plt.title("Linear model mit Aussreissern")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Data', 'Fit from P1', 'P2'])
plt.show()