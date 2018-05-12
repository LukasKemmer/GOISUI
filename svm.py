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

# set seed
np.random.seed(1001001)

# shape
m = X_training.shape[0]
n = X_training.shape[1]

# reshape y
y_training = y_training.reshape((m,))
y_test = y_test.reshape((X_test.shape[0],))

# set parameters
l = 1

# initial solution. Note: w = x[0:n], b = x[n], e = x[n+1:]
w = np.random.rand(n)
b = np.random.rand(1)
e = np.ones(m) - y_training*(np.dot(X_training, w)+b)
x0 = np.concatenate((w, b, e))

#x0 = np.concatenate((np.zeros(n+1), np.ones(m)))

# define target function
svm = lambda x : 1/m*np.sum(x[n+1:]) + l*np.sum(np.power(x[0:n], 2))

# define gradient
svm_grad = lambda x : np.concatenate((2*l*x[0:n], np.zeros(1), 1/m * np.ones((m))))

# set constraints
cons = ({'type': 'ineq', 'fun': lambda x: x[n+1:]}, # e >= 0
        {'type': 'ineq', 'fun': lambda x: -np.ones((m,)) + x[n+1:] # -1 + e -y * (Xw+b) >=0
        + y_training * (np.dot(X_training, x[0:n]) + x[n])}) # mit * als elementweiser Multiplikation

# Run optimization
res = minimize(
        fun=svm,
        x0=x0,
        method='SLSQP',
        jac=svm_grad,
        bounds=None,
        constraints=cons,
        options={'disp': True})

# extract parameters from res.x
w = res.x[0:n]
b = res.x[n]
e = res.x[n+1:]

# Make predictions
y_pred = np.dot(X_test, w) + b
y_pred[y_pred>=0] = 1
y_pred[y_pred<0] = -1

# Calculate accuracy
acc = np.sum(y_pred==y_test)/len(y_test)
print("Accuracy:",np.round(acc,2))