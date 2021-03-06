import numpy as np
from maxmin_approx import maxmin_approx
from NNarray import NN_L2


n = 2500
ns = 10
m = 30
d = 2
np.random.seed(123)
locs = np.random.rand(n, d).astype('float32')
odr = maxmin_approx(locs)
locs = locs[odr, :]
NN = NN_L2(locs, m)


"""
Unfortunately, it seems torch and faiss are not compatible 
in some cases, .e.g, under certain versions, OS
So torch here is imported after NN array is constructed
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from fit_map import fit_map_mini, compute_scal, cond_samp


torch.manual_seed(1)
locs = torch.from_numpy(locs)
NN = torch.from_numpy(NN)[:, 1:]
covM = torch.exp(-torch.cdist(locs, locs).div(2)) + torch.eye(n)
distObj = MultivariateNormal(torch.zeros(n), covM)
data = distObj.sample(torch.Size([ns]))


scal = compute_scal(locs, NN)
fitLin = fit_map_mini(data, NN, linear=True, scal=scal, lr=1e-4)
fitNonlin = fit_map_mini(data, NN, linear=False, scal=scal, lr=1e-4)


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