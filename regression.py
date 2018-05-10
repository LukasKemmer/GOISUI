#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 23:22:51 2018

@author: lukaskemmer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

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

## P1 (keine analytische Lösung)
# set c, A and b for optimization problem of form: min c^T x s.t. Ax<=b
# where x is vector (alpha a b )^T
c = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[1])))
A = np.row_stack((np.column_stack((-np.eye(X.shape[0]), X)),
                   np.column_stack((-np.eye(X.shape[0]), -X))))
b = np.row_stack((y, -y))

# solve linear problem with simplex
res = linprog(c, A_ub=A, b_ub=b, options={"disp": True}, bounds = (None, None))
beta_p1 = res.x[-2:]

## P2 (beta = (X^T*X)^-1*X^T*Y)
beta_p2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)

##########################################################
# c) Plotte Datenpunkte sowie f1, f2
##########################################################

plt.style.use(['seaborn-talk'])
fig, ax = plt.subplots()
ax.plot(X[:,0], y, 'o')
ax.plot(X[:,0], beta_p1[0]*X[:,0]+beta_p1[1])
ax.plot(X[:,0], beta_p2[0]*X[:,0]+beta_p2[1])
plt.title("Linear model ohne Aussreisser")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Data', 'Fit from P1', 'Fit from P2'])
plt.show()

##########################################################
# d) Datenpunkte hinzufügen und b), c) wiederholen
##########################################################

# Fuege Ausreisser hinzu
X = np.row_stack((X, np.array([[1, 1], [2, 1]])))
y = np.row_stack((y, np.array([[30], [35]])))

## P1 (keine analytische Lösung)
# set c, A and b for optimization problem of form: min c^T x s.t. Ax<=b
# where x is vector (alpha a b )^T
c = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[1])))
A = np.row_stack((np.column_stack((-np.eye(X.shape[0]), X)),
                   np.column_stack((-np.eye(X.shape[0]), -X))))
b = np.row_stack((y, -y))

# solve linear problem with simplex
res = linprog(c, A_ub=A, b_ub=b, options={"disp": True}, bounds = (None, None))
beta_p1 = res.x[-2:]

# P2 (beta = (X^T*X)^-1*X^T*Y)
beta_p2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)

fig, ax = plt.subplots()
ax.plot(X[:,0], y, 'o')
ax.plot(X[:,0], beta_p1[0]*X[:,0]+beta_p1[1])
ax.plot(X[:,0], beta_p2[0]*X[:,0]+beta_p2[1])
plt.title("Linear model mit Aussreissern")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Data', 'Fit from P1', 'Fit from P2'])
plt.show()