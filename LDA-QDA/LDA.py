#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
from numpy.linalg import svd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from functools import reduce

datastes = load_breast_cancer()
# print(datastes)
X = datastes['data']
y = datastes['target']

# print(X.shape)
# print(y)

idx0, idx1 = y==0, y==1
X0, X1 = X[idx0], X[idx1]
y0, y1 = y[idx0], y[idx1]

mu0 = np.mean(X0, axis=0)
mu1 = np.mean(X1, axis=0)

# cov0 = np.cov(X0.T)
# cov1 = np.cov(X1.T)
delta0 = X0-mu0
delta1 = X1-mu1
cov0 = np.dot(delta0.T, delta0)
cov1 = np.dot(delta1.T, delta1)
dr = 2

Sw = cov0 + cov1
U, sigma, VT = svd(Sw)
Sw_inverse = reduce(np.dot, [VT.T, sigma, U.T])
Omega = np.dot(Sw_inverse, (mu0-mu1))

X_reduce = np.dot(Omega, X)

print(X_reduce.shape)
