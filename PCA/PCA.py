#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
from numpy.linalg import eig
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

datasets = load_iris()
X = datasets['data']
y = datasets['target']
m, d = X.shape

cent = np.mean(X, axis=0)
X2cent = X - cent
covarX = np.dot(X2cent, X2cent.T)
e_vals, e_vecs = eig(covarX)
d_r = 2
idx = np.argsort(-e_vals) < d_r
vals = e_vals[idx]
Lambda = np.diag(vals)
vecs = e_vecs[:, idx]
X_reduction = np.dot(vecs, Lambda**.5 )

print('the projection matrix:', vecs)
print('the position in new space:', X_reduction)

pca = PCA(n_components=2)
X_reduction2 = pca.fit_transform(X)

print('using sklearn pca:', X_reduction)