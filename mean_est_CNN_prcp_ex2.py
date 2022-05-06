import numpy as np
import matplotlib.pyplot as plt
from NNVecchia import *
from torch import nn

# data processing
data = np.genfromtxt("data/prec.csv", delimiter=',', dtype='float32')[:, 1:]
lonlat = np.genfromtxt("data/lonlat.csv", delimiter=',', dtype='float32')[:, 1:]
d = 3
n = data.shape[1]
ns = data.shape[0]-3
locs = data[:d, :]
data = data[d:, :]
torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
lonlat = torch.from_numpy(lonlat)
lon = torch.unique(lonlat[:, 0]).sort().values
lat = torch.unique(lonlat[:, 1]).sort().values
indLonKnow = torch.arange(0, lon.size(dim=0), 4)
indLonKnow = torch.cat((indLonKnow, torch.tensor([lon.size(dim=0) - 1])), 0)
indLatKnow = torch.arange(0, lat.size(dim=0), 4)
indLatKnow = torch.cat((indLatKnow, torch.tensor([lat.size(dim=0) - 1])), 0)
data = data.reshape(ns, 192, 288)
locs = locs.reshape(d, 192, 288)
lenLon = 4
lenLat = 4
indLonUnknow = torch.from_numpy(
    np.setdiff1d(np.arange(indLonKnow[lenLon // 2 - 1] + 1,
                           indLonKnow[- lenLon // 2], 1),
                 indLonKnow.numpy()))
indLatUnknow = torch.from_numpy(
    np.setdiff1d(np.arange(indLatKnow[lenLat // 2 - 1] + 1,
                           indLatKnow[- lenLat // 2], 1),
                 indLatKnow.numpy()))


# find response on the grid
def rep_on_grid(iLons, iLats, iTimes, indLonKnow, indLatKnow, data, locs,
                lenLon = 4, lenLat = 4):
    batsz = iLons.size(dim=0)
    reps = torch.zeros(batsz, 4, lenLon, lenLat)
    for i in range(batsz):
        iLon = (indLonKnow < iLons[i]).sum()
        iLat = (indLatKnow < iLats[i]).sum()
        reps[i, 0, :, :] = data[np.ix_([iTimes[i]],
                                    indLatKnow[iLat - int(lenLon / 2):
                                               iLat + int(lenLon / 2)],
                                    indLonKnow[iLon - int(lenLat / 2):
                                               iLon + int(lenLat / 2)])].squeeze()
        reps[i, 1:, :, :] = locs[np.ix_(torch.arange(3),
                                       indLatKnow[iLat - int(lenLon / 2):
                                                  iLat + int(lenLon / 2)],
                                       indLonKnow[iLon - int(lenLat / 2):
                                                  iLon + int(lenLat / 2)])] - \
                            locs[:, iLats[i]:iLats[i]+1, iLons[i]:iLons[i]+1].\
                                expand(-1, lenLat, lenLon)
    return reps


# build CNN
modules = []
modules.append(nn.Conv2d(1 + d, 16, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Conv2d(16, 4, 3, padding='same'))
modules.append(nn.ReLU())
modules.append(nn.Flatten())
mdl1 = nn.Sequential(*modules)
modules = []
modules.append(nn.Linear(4 * lenLat * lenLon + 2, 64))
modules.append(nn.ReLU())
modules.append(nn.Linear(64, 8))
modules.append(nn.ReLU())
modules.append(nn.Linear(8, 1))
modules.append(nn.ReLU())
mdl2 = nn.Sequential(*modules)

# training
maxEpoch = 20
batsz = 1024
epochIter = int(n / batsz)
lr = 1e-4
optimizer = torch.optim.Adam([{'params': mdl1.parameters()},
                              {'params': mdl2.parameters()}], lr=lr)
lossFunc = torch.nn.MSELoss()
for i in range(maxEpoch):
    for j in range(epochIter):
        # Build the input tensor
        with torch.no_grad():
            indLon = torch.from_numpy(np.random.choice(indLonUnknow, batsz))
            indLat = torch.from_numpy(np.random.choice(indLatUnknow, batsz))
            indTime = torch.multinomial(torch.ones(ns - 1), batsz, True)
            input1 = rep_on_grid(indLon, indLat, indTime, indLonKnow, indLatKnow, data, locs,
                                 lenLon, lenLat)
            input2 = torch.cat((lon[indLon, None], lat[indLat, None]), 1)
            yTrue = data[indTime, indLat, indLon]
        optimizer.zero_grad()
        output1 = mdl1(input1)
        input2 = torch.cat((output1, input2), 1)
        yhat = mdl2(input2).squeeze()
        loss = lossFunc(yhat, yTrue)
        loss.backward()
        optimizer.step()
    print("Epoch", i)
    print("Loss is", loss.item())

with torch.no_grad():
    indLon = torch.kron(indLonUnknow, torch.ones(indLatUnknow.size(dim=0))).type(torch.long)
    indLat = torch.kron(torch.ones(indLonUnknow.size(dim=0)), indLatUnknow).type(torch.long)
    indTime = (torch.ones(indLat.size(dim=0))*(ns-1)).type(torch.long)
    input1 = rep_on_grid(indLon, indLat, indTime, indLonKnow, indLatKnow, data, locs,
                         lenLon, lenLat)
    input2 = torch.cat((lon[indLon, None], lat[indLat, None]), 1)
    yTrue = data[indTime, indLat, indLon]
    output1 = mdl1(input1)
    input2 = torch.cat((output1, input2), 1)
    yhat = mdl2(input2)

print(f"RMSE is {np.sqrt(np.square(yhat - yTrue).mean())}") # 0.40634083
yPred = data[ns-1].detach().clone()
yPred[np.ix_(indLatUnknow, indLonUnknow)] = yhat.reshape(indLatUnknow.size(dim=0),
                                                 indLonUnknow.size(dim=0))
yTrue = data[ns-1].numpy()
plt.imshow(yPred)
plt.imshow(yTrue)



