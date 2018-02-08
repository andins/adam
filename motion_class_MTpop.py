#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 07:50:40 2018

@author: andrea
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set_context('poster')

data = loadmat('/home/andrea/Work/adam_morris/data/MTarraySpks.mat')
single_units = data['data']['su'][0][0]  # nNeurons x nTrials x nBins
multi_unit = data['data']['mu'][0][0]  # nChannels x nTrials x nBins
motion_dir = data['data']['dir'][0][0][0]
motion_speed = data['data']['speed'][0][0][0]
stimulus_on = data['data']['stimOn'][0][0][0]  # nBins ones when stimulus is on
# average activity over stimulus period (shifted 50 ms to account for latency)
#avg_activity_stim = single_units[:, :, 50:550].sum(axis=2)
avg_activity_stim = multi_unit[:, :, 50:550].sum(axis=2)

X = avg_activity_stim.T
y = motion_speed
pca = PCA()
pca.fit(X)
X_PCspace = pca.transform(X)
rho = np.zeros([X_PCspace.shape[1]])
p = np.zeros([X_PCspace.shape[1]])
for d in range(X_PCspace.shape[1]):
    rho[d], p[d] = pearsonr(X_PCspace[:, d], y)
plt.figure()
plt.plot(rho, linewidth=2)
plt.plot(p, linewidth=2)
# plot PCA spectrum
plt.figure()
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance')
# plot correlation matrix of features
corr_matrix = np.corrcoef(X.T)
np.fill_diagonal(corr_matrix, 0)
plt.figure()
sns.heatmap(corr_matrix)
# plot data in the space of 3 PCs (with highest correlation with y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sns.set_palette('hls')
for dd in np.unique(y):
    ax.scatter(X_PCspace[y == dd, 0], X_PCspace[y == dd, 1],
               X_PCspace[y == dd, 2])
plt.figure()
for dd in np.unique(y):
    plt.scatter(X_PCspace[y == dd, 1], X_PCspace[y == dd, 2])

# f-score between each neuron and y
selector = SelectKBest(f_classif, k=X.shape[1])
selector.fit(X, y)
plt.figure()
plt.plot(selector.scores_)
# classification
kf = KFold(n_splits=10)
zscore = StandardScaler()
clf = LogisticRegression(C=.1, multi_class='multinomial', solver='lbfgs')
#clf = SVC(kernel='rbf')

pipe = Pipeline([('pca', pca), ('clf', clf)])
pipe.set_params(pca__n_components=X.shape[1])
scores = np.zeros([10])
f = 0
for train, test in kf.split(X):
    pipe.fit(X[train, :], y[train])
    scores[f] = pipe.score(X[test, :], y[test])
    f += 1
print("CV accuracy: %.2f +/- %.3f" %(scores.mean(), scores.std()))
