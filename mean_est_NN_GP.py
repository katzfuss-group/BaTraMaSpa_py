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
from NNVecchia import build_NNVecc
from torch.distributions.multivariate_normal import MultivariateNormal


torch.manual_seed(1)
locs = torch.from_numpy(locs)
NN = torch.from_numpy(NN)[:, 1:]
covM = torch.exp(-torch.cdist(locs, locs).div(2)) + torch.eye(n)
distObj = MultivariateNormal(torch.zeros(n), covM)
data = distObj.sample(torch.Size([ns]))
ySlice = data[0]
locsNy = torch.cat((locs, ySlice.unsqueeze(1)), 1)
nOOS = 500

lr = 1e-4
maxEpoch = 30
batsz = 128
epochIter = int(n / batsz)
mdlNNVecc = build_NNVecc(m, d, m * d, m * d, m * d, m * d, m * d, m * d)
optimizer = torch.optim.Adam(mdlNNVecc.parameters(), lr=lr)
lossFunc = torch.nn.MSELoss()
for i in range(maxEpoch):
    for j in range(epochIter):
        inds = torch.multinomial(torch.ones(n - m - nOOS), batsz) + m
        # Build the input tensor
        input = torch.zeros([batsz, d * (m + 1) + m])
        input[:, :d] = locs[inds, :]
        for k in range(batsz):
            input[k, d:] = torch.flatten(locsNy[NN[inds[k], :], :])
        optimizer.zero_grad()
        yPred = mdlNNVecc(input).squeeze()
        yBat = ySlice[inds]
        loss = lossFunc(yPred, yBat)
        loss.backward()
        optimizer.step()
    print(f"Epoch {i} Finished")
    print(f"Loss is {loss.item()}")

with torch.no_grad():
    inds = torch.arange(n - nOOS, n)
    input = torch.zeros([nOOS, d * (m + 1) + m])
    input[:, :d] = locs[inds, :]
    for k in range(nOOS):
        input[k, d:] = torch.flatten(locsNy[NN[inds[k], :], :])
    yPred = mdlNNVecc(input).squeeze()
    yBat = ySlice[inds]
    loss = lossFunc(yPred, yBat)
    print("NN loss is", loss.item())
    yTmp = torch.linalg.solve(covM[:n-nOOS, :n-nOOS], ySlice[:n-nOOS])
    yPredGP = torch.mm(covM[n-nOOS:, :n-nOOS], yTmp.unsqueeze(1)).squeeze()
    loss = lossFunc(yPredGP, yBat)
    print("GP loss is", loss.item())











