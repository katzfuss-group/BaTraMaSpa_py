import numpy as np
from maxmin_approx import maxmin_approx
from NNarray_sklearn import NN_L2
from NNVecchia import *
from torch import nn
import matplotlib.pyplot as plt

# data processing, maxmin order and NN search
# data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
data = np.genfromtxt("data/temp.csv", delimiter=',', dtype='float32')
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
torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
NN = torch.from_numpy(NN)

# build NN
# Use coordinate differences and CNN to predict the weights for kriging
modules = []
modules.append(nn.Conv1d(d, 10, 5, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d(10, 10, 5, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d(10, 10, 5, padding='same'))
modules.append(nn.ReLU())
# modules.append(nn.Dropout(0.2))
modules.append(nn.Flatten())
modules.append(nn.Linear(10 * m, 10 * m))
modules.append(nn.ReLU())
modules.append(nn.Linear(10 * m, 10 * m))
modules.append(nn.ReLU())
modules.append(nn.Linear(10 * m, m))
mdlNNVecc = nn.Sequential(*modules)

# train NN
maxEpoch = 30
batsz = 1024
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
            input = torch.zeros([batsz, d, m])
            obs = torch.zeros([batsz, m])
            yTrue = torch.zeros(batsz)
            for k in range(batsz):
                input[k, :, :] = (locs[NN[indLoc[k], 1:], :] -
                                  locs[NN[indLoc[k], :1], :].expand(m, -1)).\
                        transpose(0, 1)
                obs[k, :] = data[indTime[k], NN[indLoc[k], 1:]]
                yTrue[k] = data[indTime[k], indLoc[k]]
        optimizer.zero_grad()
        weight = mdlNNVecc(input).squeeze()
        yhat = (weight * obs).sum(dim=-1)
        loss = lossFunc(yhat, yTrue)
        loss.backward()
        optimizer.step()
    print("Epoch", i)
    print("Loss is", loss.item())

# predict using OOS dataset
nKnown = max(int(n * 0.1), m)
with torch.no_grad():
    X = torch.zeros([1, d, m])
    yhat = torch.zeros(n)
    yhat[:nKnown] = data[ns-1, :nKnown]
    for k in range(nKnown, n):
        X[0, :, :] = (locs[NN[k, 1:], :] - locs[NN[k, :1], :].expand(m, -1)).transpose(0, 1)
        obs = yhat[NN[k, 1:]]
        weight = mdlNNVecc(X).squeeze()
        yhat[k] = (weight * obs).sum()

# plot predicted map and true map
i = 0
odrInv = np.arange(odr.size)
odrInv[odr] = np.arange(odr.size)
ySim = yhat.numpy()[odrInv].reshape(torch.unique(locs[:, 2]).size(dim=0),
                                    locs.size(dim=0) //
                                    torch.unique(locs[:, 2]).size(dim=0))
yTrue = data[ns-1, :].numpy()[odrInv].reshape(torch.unique(locs[:, 2]).size(dim=0),
                                              locs.size(dim=0) //
                                              torch.unique(locs[:, 2]).size(dim=0))
plt.imshow(ySim)
plt.imshow(yTrue)
print(f"RMSE is {np.sqrt(np.square(ySim - yTrue).mean())}") # 0.3984974324703216


