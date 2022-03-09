import torch
import numpy as np
from numpy import genfromtxt
import fit_map

NN = torch.from_numpy(genfromtxt(
    '/Users/caoj/Documents/BaTraMaSpa/code/NNarray_max.csv',
    delimiter=' ', missing_values='NA', dtype=np.int)).sub(1)
locs = torch.from_numpy(genfromtxt(
    '/Users/caoj/Documents/BaTraMaSpa/code/locs_ord.csv',
    delimiter=' ', missing_values='NA', dtype=np.float32))
data = torch.from_numpy(genfromtxt(
    '/Users/caoj/Documents/BaTraMaSpa/code/data_all.csv',
    delimiter=' ', missing_values='NA', dtype=np.float32)).\
    transpose(-2, -1)

scal = fit_map.compute_scal(locs, NN)
fitLin = fit_map.fit_map_mini(data, NN, False, scal=scal, lr=1e-4, maxIter=5)
fitNonlin = fit_map.fit_map_mini(data, NN, False, scal=scal, lr=1e-4, maxIter=5)

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
            fx[k, l] = fit_map.cond_samp(fitNonlin, 'fx', xFix=xFix, indLast=i)[i]
            fxLin[k, l] = fit_map.cond_samp(fitLin, 'fx', xFix=xFix, indLast=i)[i]
