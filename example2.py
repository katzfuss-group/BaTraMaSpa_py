import numpy as np
from maxmin_approx import maxmin_approx
from NNarray import NN_L2


data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
d = 3
n = data.shape[1]
ns = data.shape[0]-3
m = 30

locs = np.transpose(data[:d, :])
data = data[d:, :]
np.random.seed(123)
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
from NNVecchia import *


torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
NN = torch.from_numpy(NN)

pIn = 1
z = torch.normal(0.0, 1.0, data.shape)
G = build_gen(m*d+m+d, 1, 1, m*d*4, m*d*3, m*d*2, m*d*1)
D = build_dis(m*d+m+d+1, m*d*4, m*d*3, m*d*2, m*d*1)
maxEpoch = 50
batsz = 128
batszHalf = int(batsz / 2)
epochIter = int(n / batsz)
lr = 1e-4
optimGen = torch.optim.Adam(G.parameters(), lr=lr)
optimDis = torch.optim.Adam(D.parameters(), lr=lr)
for i in range(maxEpoch):
    for j in range(epochIter):
        # Build the input tensor
        with torch.no_grad():
            indLoc = torch.multinomial(torch.ones(n - m), batsz, False) + m
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            X1 = torch.zeros([batsz, (d+1)*(m+1)])
            for k in range(batsz):
                X1[k, :d*(m+1)] = torch.flatten(locs[NN[indLoc[k], :], :])
                X1[k, d*(m+1):m*d+m+d] = data[indTime[k], NN[indLoc[k], 1:]]
                X1[k, m*d+m+d] = z[indTime[k], indLoc[k]]
            yhat = G(X1[:batszHalf]).squeeze()
            for k in range(batszHalf, batsz):
                X1[k, m*d+m+d] = data[indTime[k], indLoc[k]]
            X1[:batszHalf, m*d+m+d] = yhat
        optimDis.zero_grad()
        p = D(X1).squeeze()
        lossDis = -(torch.sum(p[:batszHalf]) - torch.sum(torch.exp(p[batszHalf:])))
        lossDis.backward()
        optimDis.step()

        with torch.no_grad():
            indLoc = torch.multinomial(torch.ones(n - m), batsz, False) + m
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            X2 = torch.zeros([batsz, (d + 1) * (m + 1)])
            for k in range(batsz):
                X2[k, :d*(m+1)] = torch.flatten(locs[NN[indLoc[k], :], :])
                X2[k, d*(m+1):m*d+m+d] = data[indTime[k], NN[indLoc[k], 1:]]
                X2[k, m*d+m+d] = z[indTime[k], indLoc[k]]
        optimGen.zero_grad()
        yhat = G(X2).squeeze()
        X3 = X2.detach().clone()
        X3[:, m*d+m+d] = yhat
        p = D(X3)
        lossGen = torch.sum(p)
        lossGen.backward()
        optimGen.step()
    print("Epoch", i)
    print("Discriminator loss is", lossDis.item())
    print("Generator loss is", lossGen.item())

nKnown = max(int(n * 0.1), m)
nSim = 1000
with torch.no_grad():
    X = torch.zeros([nSim, (d + 1) * (m + 1)])
    yhat = torch.zeros([nSim, n])
    yhat[:, :nKnown] = data[ns-1:, :nKnown].expand(nSim, -1)
    for k in range(nKnown, n):
        X[:, :d * (m + 1)] = torch.flatten(locs[NN[k, :], :]).unsqueeze(0).expand(nSim, -1)
        X[:, d * (m + 1):m * d + m + d] = yhat[:, NN[k, 1:]]
        X[:, m * d + m + d] = torch.normal(0.0, 1.0, (nSim,))
        yhat[:, k] = G(X).squeeze()

def inv_odr(odr):
    odrInv = np.zeros(odr.size)
    for i, p in enumerate(odr):
        odrInv[p] = i
    return odrInv

import matplotlib.pyplot as plt
i = 0
odrInv = inv_odr(odr).astype('int')
ySim = yhat.numpy()[i, odrInv].reshape(192, 288)
yTrue = data[ns-1, :].numpy()[odrInv].reshape(192, 288)
plt.imshow(ySim)
plt.imshow(yTrue)

np.savetxt("true_and_sim.csv",
           torch.cat((data[ns-1:ns, odrInv],
                      yhat[:, odrInv]), 0).transpose(0, 1).numpy(),
           delimiter=",")


