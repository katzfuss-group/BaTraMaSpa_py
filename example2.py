
import numpy as np
from maxmin_approx import maxmin_approx
from NNarray import NN_L2

data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
d = 3
n = data.shape[1]
ns = data.shape[0] - 3
m = 30

locs = np.transpose(data[:d, :])
data = data[d:, :]
odr = maxmin_approx(locs)
locs = locs[odr, :]
data = data[:, odr]
NN = NN_L2(locs, m)


"""
Unfortunately, it seems torch and faiss are not compatible 
in some cases, .e.g, under certain versions, OS
So torch here is imported after NN array is constructed
"""

import torch
from fit_map import fit_map_mini, compute_scal, cond_samp


torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
NN = torch.from_numpy(NN)[:, 1:]


scal = compute_scal(locs, NN)
fitLin = fit_map_mini(data, NN, True, scal=scal, lr=1e-4, maxIter=100)
fitNonlin = fit_map_mini(data, NN, False, scal=scal, lr=1e-4, maxIter=100)


i = 79
NNrow = NN[i, :]
xFix = torch.zeros(i)
xFix[NNrow] = data[:, NNrow].mean(dim=0)
nVal = 20
NNVal = torch.zeros(2, nVal)
NNVal[0, :] = torch.linspace(start=data[:, NNrow[0]].min(dim=0).values,
                             end=data[:, NNrow[0]].max(dim=0).values,
                             steps=nVal)
NNVal[1, :] = torch.linspace(start=data[:, NNrow[1]].min(dim=0).values,
                             end=data[:, NNrow[1]].max(dim=0).values,
                             steps=nVal)
fx = torch.zeros(nVal, nVal)
fxLin = torch.zeros(nVal, nVal)
with torch.no_grad():
    for k in range(nVal):
        for l in range(nVal):
            xFix[NNrow[:2]] = torch.tensor([NNVal[0, k], NNVal[1, l]])
            fx[k, l] = cond_samp(fitNonlin, 'fx', xFix=xFix, indLast=i)[i]
            fxLin[k, l] = cond_samp(fitLin, 'fx', xFix=xFix, indLast=i)[i]


import matplotlib.pyplot as plt


fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(NNVal[0, :], NNVal[1, :])
ax.scatter3D(X, Y, fxLin, c=fxLin, cmap='Greens')
ax.set_xlabel('1st NN')
ax.set_ylabel('2nd NN')


fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(NNVal[0, :], NNVal[1, :])
ax.scatter3D(X, Y, fx, c=fx, cmap='Greens')
ax.set_xlabel('1st NN')
ax.set_ylabel('2nd NN')