import numpy as np
from NNVecchia import *
from torch import nn
from maxmin_approx import maxmin_approx

# data processing
data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
d = 3
n = data.shape[1]
ns = data.shape[0]-3
locs = np.transpose(data[:d, :])
data = data[d:, :]
torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
lat = torch.unique(locs[:, 2])

# build two CNNs as generator and discriminator
nLon = 288
nLat = 192
nZ = 50
modules = []
modules.append(nn.Linear(nZ + 1, 64))
modules.append(nn.ReLU())
modules.append(nn.Linear(64, 64))
modules.append(nn.ReLU())
modules.append(nn.Linear(64, 64))
modules.append(nn.ReLU())
modules.append(nn.Linear(64, nLon))
G = nn.Sequential(*modules)

modules = []
modules.append(nn.Conv1d(1, 64, 5))
modules.append(nn.ReLU())
modules.append(nn.Conv1d(64, 32, 5))
modules.append(nn.ReLU())
modules.append(nn.Conv1d(32, 16, 5))
modules.append(nn.ReLU())
modules.append(nn.Dropout(0.2))
modules.append(nn.Flatten())
modules.append(nn.Linear((nLon - 3 * 4) * 16, 1))
D = nn.Sequential(*modules)

# model training
maxEpoch = 30
batsz = 128
epochIter = int(n / batsz)
lr = 1e-4
decay = 1e-4
optimGen = torch.optim.Adam(G.parameters(), lr=lr, weight_decay=decay)
optimDis = torch.optim.Adam(D.parameters(), lr=lr, weight_decay=decay)
for i in range(maxEpoch):
    for j in range(epochIter):
        # Build the input tensor
        with torch.no_grad():
            indLat = torch.multinomial(torch.ones(nLat), batsz, True)
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            X1 = torch.zeros([batsz, nZ+1])
            X1[:, 0] = lat[indLat]
            X1[:, 1:] = torch.normal(0.0, 1.0, (batsz, nZ))
            yhat = G(X1).squeeze()
            X1 = torch.zeros([batsz, nLon])
            for k in range(batsz):
                X1[k, :] = data[indTime[k], nLon * indLat[k]:nLon * indLat[k] + nLon]
            X1 = torch.cat((yhat, X1), dim=0)
            X1 = X1.unsqueeze(1)
        optimDis.zero_grad()
        p = D(X1).squeeze()
        lossDis = -(torch.sum(p[:batsz]) - torch.sum(torch.exp(p[batsz:])))
        lossDis.backward()
        optimDis.step()
        # Build the input tensor
        with torch.no_grad():
            indLat = torch.multinomial(torch.ones(nLat), batsz, True)
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            X1 = torch.zeros([batsz, nZ + 1])
            X1[:, 0] = lat[indLat]
            X1[:, 1:] = torch.normal(0.0, 1.0, (batsz, nZ))
        optimGen.zero_grad()
        yhat = G(X1).unsqueeze(1)
        p = D(yhat)
        lossGen = torch.sum(p)
        lossGen.backward()
        optimGen.step()
    print("Epoch", i)
    print("Discriminator loss is", lossDis.item())
    print("Generator loss is", lossGen.item())

# sample from 'marginal' distribution
nSim = 1000
with torch.no_grad():
    X = torch.zeros([nSim, nZ + 1])
    X[:, 0] = lat[95]
    X[:, 1:] = torch.normal(0.0, 1.0, (nSim, nZ))
    yhat = G(X).squeeze()
    np.savetxt("true_and_sim.csv",
               torch.cat((data[ns-1:ns, nLon * 95:nLon*95+nLon],
                          yhat), 0).transpose(0, 1).numpy(),
               delimiter=",")


