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
from NNVecchia import *
from torch import nn

torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
NN = torch.from_numpy(NN)

modules = []
modules.append(nn.Conv1d(d+1, 20, 3, padding='valid'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d(20, 20, 3, padding='valid'))
modules.append(nn.MaxPool1d(2))
modules.append(nn.Dropout(0.2))
modules.append(nn.Flatten())
modules.append(nn.Linear(260, 100))
modules.append(nn.ReLU())
modules.append(nn.Linear(100, 100))
modules.append(nn.ReLU())
modules.append(nn.Linear(100, 1))
mdlNNVecc = nn.Sequential(*modules)
maxEpoch = 30
batsz = 128
epochIter = int(n / batsz)
lr = 1e-4
optimizer = torch.optim.Adam(mdlNNVecc.parameters(), lr=lr)
lossFunc = torch.nn.MSELoss()
for i in range(maxEpoch):
    for j in range(epochIter):
        # Build the input tensor
        with torch.no_grad():
            indLoc = torch.multinomial(torch.ones(n - m), batsz, False) + m
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            input = torch.zeros([batsz, d+1, m])
            yTrue = torch.zeros(batsz)
            for k in range(batsz):
                input[k, :d, :] = (locs[NN[indLoc[k], 1:], :] -
                                   locs[NN[indLoc[k], :1], :].expand(m, -1)).transpose(0, 1)
                input[k, d, :] = data[indTime[k], NN[indLoc[k], 1:]]
                yTrue[k] = data[indTime[k], indLoc[k]]
        optimizer.zero_grad()
        yhat = mdlNNVecc(input).squeeze()
        loss = lossFunc(yhat, yTrue)
        loss.backward()
        optimizer.step()
    print("Epoch", i)
    print("Loss is", loss.item())

nKnown = max(int(n * 0.1), m)
with torch.no_grad():
    X = torch.zeros([1, d + 1, m])
    yhat = torch.zeros(n)
    yhat[:nKnown] = data[ns-1, :nKnown]
    for k in range(nKnown, n):
        X[0, :d, :] = (locs[NN[k, 1:], :] -
                       locs[NN[k, :1], :].expand(m, -1)).transpose(0, 1)
        X[0, d, :] = yhat[NN[k, 1:]]
        yhat[k] = mdlNNVecc(X).squeeze()

def inv_odr(odr):
    odrInv = np.zeros(odr.size)
    for i, p in enumerate(odr):
        odrInv[p] = i
    return odrInv

import matplotlib.pyplot as plt
i = 0
odrInv = inv_odr(odr).astype('int')
ySim = yhat.numpy()[odrInv].reshape(192, 288)
yTrue = data[ns-1, :].numpy()[odrInv].reshape(192, 288)
plt.imshow(ySim)
plt.imshow(yTrue)
print(f"RMSE is {np.sqrt(np.square(ySim - yTrue).mean())}") # 0.40634083


