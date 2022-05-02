import numpy as np
from NNVecchia import *
from torch import nn
from maxmin_approx import maxmin_approx

# data processing, maxmin order and NN search
data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
d = 3
n = data.shape[1]
ns = data.shape[0]-3
locs = np.transpose(data[:d, :])
data = data[d:, :]
torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)

# build two CNNs as generator and discriminator
nGen = 50
modules = []
modules.append(nn.Conv1d(d + 1, (d + 1) * 3, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d((d + 1) * 3, (d + 1) * 2, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d((d + 1) * 2, d + 1, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Dropout(0.2))
modules.append(nn.Flatten())
modules.append(nn.Linear((d + 1) * nGen, nGen))
G = nn.Sequential(*modules)

modules = []
modules.append(nn.Conv1d(d + 1, (d + 1) * 3, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d((d + 1) * 3, (d + 1) * 2, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv1d((d + 1) * 2, d + 1, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Dropout(0.2))
modules.append(nn.Flatten())
modules.append(nn.Linear((d + 1) * nGen, 1))
D = nn.Sequential(*modules)

# model training
z = torch.normal(0.0, 1.0, data.shape)
maxEpoch = 30
batsz = 1024
batszHalf = int(batsz / 2)
epochIter = int(n / batsz)
lr = 1e-4
optimGen = torch.optim.Adam(G.parameters(), lr=lr)
optimDis = torch.optim.Adam(D.parameters(), lr=lr)
for i in range(maxEpoch):
    for j in range(epochIter):
        # Build the input tensor
        with torch.no_grad():
            indLoc = torch.multinomial(torch.ones([batsz, n]), nGen, False)
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            X1 = torch.zeros([batsz, (d+1), nGen])
            for k in range(batsz):
                X1[k, :d, :] = locs[indLoc[k, :], :].transpose(0, 1)
                X1[k, d, :] = z[indTime[k], indLoc[k, :]]
            yhat = G(X1[:batszHalf, :, :]).squeeze()
            X1[:batszHalf, d, :] = yhat
            for k in range(batszHalf, batsz):
                X1[k, d, :] = data[indTime[k], indLoc[k, :]]
        optimDis.zero_grad()
        p = D(X1).squeeze()
        lossDis = -(torch.sum(p[:batszHalf]) - torch.sum(torch.exp(p[batszHalf:])))
        lossDis.backward()
        optimDis.step()
        # Build the input tensor
        with torch.no_grad():
            indLoc = torch.multinomial(torch.ones([batsz, n]), nGen, False)
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            for k in range(batsz):
                X1[k, :d, :] = locs[indLoc[k, :], :].transpose(0, 1)
                X1[k, d, :] = z[indTime[k], indLoc[k, :]]
        optimGen.zero_grad()
        yhat = G(X1).squeeze()
        X2 = X1.detach().clone()
        X2[:, d, :] = yhat
        p = D(X2)
        lossGen = torch.sum(p)
        lossGen.backward()
        optimGen.step()
    print("Epoch", i)
    print("Discriminator loss is", lossDis.item())
    print("Generator loss is", lossGen.item())

# sample from 'marginal' distribution
nSim = 1000
odr = maxmin_approx(locs)
with torch.no_grad():
    X = torch.zeros([nSim, d+1, nGen])
    X[:, :d, :] = locs[odr[:nGen], :].transpose(0, 1).\
        unsqueeze(0).expand(nSim, -1, -1)
    X[:, d, :] = torch.normal(0.0, 1.0, (nSim, nGen))
    yhat = G(X).squeeze()
    np.savetxt("true_and_sim.csv",
               torch.cat((data[ns-1:ns, odr[:nGen]],
                          yhat), 0).transpose(0, 1).numpy(),
               delimiter=",")


