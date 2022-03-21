# BaTraMaSpa_py
The python translation of the BaTraMaSpa repository


# Main functions

* fit_map_mini: fit linear or non-linear transport maps with mini-batch subsampling
* cond_samp: conditional sampling 

# Examples

## Example 1

Fit GP data from an exponential kernel.
```
import numpy as np
from maxmin_approx import maxmin_approx
from NNarray import NN_L2
```

Simulate locations.
```
n = 2500
ns = 10
m = 30
d = 2
locs = np.random.rand(n, d).astype('float32')
odr = maxmin_approx(locs)
locs = locs[odr, :]
NN = NN_L2(locs, m)
```

Unfortunately, it seems torch and faiss are not compatible in some cases, .e.g, under certain versions, OS. So torch here is imported after NN array is constructed. Now we simulate the GP data.
```
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from fit_map import fit_map_mini, compute_scal, cond_samp
torch.manual_seed(1)
locs = torch.from_numpy(locs)
NN = torch.from_numpy(NN)[:, 1:]
covM = torch.exp(-torch.cdist(locs, locs).div(2)) + torch.eye(n)
distObj = MultivariateNormal(torch.zeros(n), covM)
data = distObj.sample(torch.Size([ns]))
```

Fit the transport map, assuming either linear or non-linear.
```
scal = compute_scal(locs, NN)
fitLin = fit_map_mini(data, NN, True, scal=scal, lr=1e-4, maxIter=50)
fitNonlin = fit_map_mini(data, NN, False, scal=scal, lr=1e-4, maxIter=50)
```

Posterior sampling for the 80-th location.
```
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
```

## Example 2

Fit the precipitation data of 20 days.
```
import numpy as np
from maxmin_approx import maxmin_approx
from NNarray import NN_L2
```

Construct locations and precipitation data.
```
data = np.genfromtxt("data/prec20.csv", delimiter=',', dtype='float32')[:, 1:]
n = data.shape[0]
ns = data.shape[1] - 2
m = 30
d = 2
locs = np.transpose(data[:d, :])
data = data[d:, :]
data = data / data.max()
odr = maxmin_approx(locs)
locs = locs[odr, :]
NN = NN_L2(locs, m)
```

Maximin order and NN array construction,
```
import torch
from fit_map import fit_map_mini, compute_scal, cond_samp
torch.manual_seed(0)
locs = torch.from_numpy(locs)
data = torch.from_numpy(data)
NN = torch.from_numpy(NN)[:, 1:]
```

Fit the transport map, assuming either linear or non-linear.
```
scal = compute_scal(locs, NN)
fitLin = fit_map_mini(data, NN, True, scal=scal, lr=1e-4, maxIter=100)
fitNonlin = fit_map_mini(data, NN, False, scal=scal, lr=1e-4, maxIter=100)
```

Posterior sampling for the 80-th location.
```
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
```

