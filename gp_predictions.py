import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

sys.path.append('../')
from BaTraMaSpa_py.fit_map import compute_scal, fit_map_mini, covar_samples
from BaTraMaSpa_py.maxmin_approx import maxmin_approx
from BaTraMaSpa_py.NNarray import NN_L2

from my_utils import construct_gp, make_uniform_grid

FIGPATH = '../figures/covariate_experiments/dec_update/12022022'

train_ls = (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95)
test_ls = (0.1, 0.25, 0.5, 0.75, 0.9)

n = 30
trainsize = 50
testsize = 6

train = construct_gp(nsamples = trainsize, length_scale = train_ls, n = n, seed = 1)
print(f"TRAINING DATA SIZE: {train.shape}")
train = train.reshape(-1, n**2)

test = construct_gp(nsamples = testsize, length_scale = test_ls, n = n, seed = 35256)
print(f"VALIDATION DATA SIZE: {test.shape}")
test = test.reshape(-1, n**2)

locs = make_uniform_grid(n = n, d = 2).astype(np.float32)

order = maxmin_approx(locs)

locs = locs[order]

NN = NN_L2(locs, 30)

locs = torch.from_numpy(locs).float()
train = torch.from_numpy(train).float()
test = torch.from_numpy(test).float()
NN = torch.from_numpy(NN)[:, 1:].long()

Xtrain = torch.tensor(train_ls).repeat_interleave(trainsize).unsqueeze(1)
Xtrain = Xtrain.mul(torch.ones(train.shape)).unsqueeze(-1).log()
Xtest = torch.tensor(test_ls).repeat_interleave(testsize).unsqueeze(1)
Xtest = Xtest.mul(torch.ones(test.shape)).unsqueeze(-1).log()

scale = compute_scal(locs, NN)

# Fitting and sampling from the map.
# Want this to be completely repeatable.
torch.manual_seed(1)
tm = fit_map_mini(train[:, order], Xtrain, NN, scal = scale, maxEpoch = 30, lr = 1e-5)

# fwd = torch.empty_like(test)
new_samples = torch.empty_like(test)
    
with torch.no_grad():
    for i in range(test.shape[0]):
        # fwd[i, order] = covar_samples(tm, mode = 'trans', Y_obs = test[i, order], X_obs = Xtest[i, order])
        new_samples[i, order] = covar_samples(tm, mode = 'bayes', X_obs = Xtest[i, order])

new_samples = new_samples.numpy()
test = test.numpy()

test = test.reshape(len(test_ls), testsize, -1)
new_samples = new_samples.reshape(len(test_ls), testsize, -1)

tmin, tmax = np.min(test), np.max(test)
fmin, fmax = np.min(new_samples), np.max(new_samples)

test_kwargs = {'cmap': 'Spectral_r', 'vmin': tmin, 'vmax': tmax}
new_kwargs = {'cmap': 'Spectral_r', 'vmin': fmin, 'vmax': fmax}

# Plotting
fig, ax = plt.subplots(testsize, len(test_ls), figsize = (12, 10), tight_layout = True)

for i, ls in enumerate(test_ls):
    for j in range(testsize):
        im0 = ax[j, i].imshow(test[i, j].reshape(n, n), **test_kwargs)
        plt.colorbar(im0, ax = ax[j, i])
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        
for i, ls in enumerate(test_ls):
    ax[0, i].set_title('LS = {}'.format(ls))

plt.savefig(FIGPATH + '/gp_samples.png', dpi = 600)

fig, ax = plt.subplots(testsize, len(test_ls), figsize = (12, 10), tight_layout = True)

for i, ls in enumerate(test_ls):
    for j in range(testsize):
        im0 = ax[j, i].imshow(new_samples[i, j].reshape(n, n), **test_kwargs)
        plt.colorbar(im0, ax = ax[j, i])
        ax[j, i].set_xticks([])
        ax[j, i].set_yticks([])
        
for i, ls in enumerate(test_ls):
    ax[0, i].set_title('LS = {}'.format(ls))

plt.savefig(FIGPATH + '/tm_samples.png', dpi = 600)

# Covar plots
with torch.no_grad():
    xthetas = tm['theta'][-2:]
    fit_scale = scale.mul(xthetas[1]).add(xthetas[0]).exp()
    Xs = Xtest.squeeze()[:, order].mul(fit_scale)[:, order.argsort()]

Xs = Xs.reshape(5, -1, n**2)

fig, ax = plt.subplots(Xs.shape[0], Xs.shape[1], figsize = (12, 5), tight_layout = True)

for i in range(Xs.shape[0]):
    for j in range(Xs.shape[1]):
        if i == 0:
            ax[0, j].set_title('LS = {}'.format(test_ls[i]))
        im = ax[i, j].imshow(Xs[i, j].numpy().reshape(n, n), cmap = 'Spectral_r', vmin = Xs.min(), vmax = Xs.max())
        plt.colorbar(im, ax = ax[i, j])
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

plt.savefig(FIGPATH + '/Covarmaps.png', dpi = 600)