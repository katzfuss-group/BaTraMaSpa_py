import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from fit_map import compute_scal, cond_samp, fit_map_mini, TransportMap
from maxmin_approx import maxmin_approx
from NNarray import NN_L2

# Fig path and files to read from a higher directory
# gp data and locs generated using example3.construct_gp
# and example3.make_uniform_grid. Files saved via torch.save.
FIGPATH = '../figures/covariate_experiments/'
FILEPATH = '../data/simulations/gp_covars/'
GP_DATA = 'gp_050208.npy'
LOCS = 'locs.npy'

def plot_experiment(sample_index, figname, tm, lengthscale):
    """
    Modification of example3.main from this repo. Plots experiments in
    using a marginal transport map only.
    """

    Y: torch.tensor = tm['Y_data']
    n = Y.size(1)
    
    scale: torch.tensor = tm['scal']

    with torch.no_grad():
        fx_samples = []
        fwd_samples = []
        inv_samples = []

        fx_min  = fx_max  = 0
        fwd_min = fwd_max = 0
        inv_min = inv_max = 0

        for i in range(6):
            z = torch.randn(n**2)
            fx_sample = cond_samp(tm, 'fx')
            fx_min = min(fx_min, fx_sample.min())
            fx_max = max(fx_max, fx_sample.max())
            fx_samples.append(fx_sample)

            fwd_sample = cond_samp(tm, 'trans', Y_obs = Y[sample_index+i])
            fwd_min = min(fwd_min, fwd_sample.min())
            fwd_max = max(fwd_max, fwd_sample.max())
            fwd_samples.append(fwd_sample)
            
            inv_sample = cond_samp(tm, 'invtrans', Y_obs = fwd_sample)
            inv_min = min(inv_min, inv_sample.min())
            inv_max = max(inv_max, inv_sample.max())
            inv_samples.append(inv_sample)

    fig, ax = plt.subplots(4, 6, figsize=(15, 6), constrained_layout = True)

    ymin = Y[sample_index:sample_index+6].min()
    ymax = Y[sample_index:sample_index+6].max()

    for i, (fx, fwd, inv) in enumerate(zip(fx_samples, fwd_samples, inv_samples)):
        im0 = ax[0, i].imshow(Y[sample_index+i].reshape(n, n), cmap = 'Spectral_r', vmin = ymin, vmax = ymax)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        plt.colorbar(im0, ax=ax[0, i])

        im1 = ax[1, i].imshow(fx.reshape(n, n), cmap = 'Spectral_r', vmin = fx_min, vmax = fx_max)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        plt.colorbar(im1, ax=ax[1, i])
        
        im2 = ax[2, i].imshow(fwd.reshape(n, n), cmap = 'Spectral_r', vmin = fwd_min, vmax = fwd_max)
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])
        plt.colorbar(im2, ax=ax[2, i])
        
        im3 = ax[3, i].imshow(inv.reshape(n, n), cmap = 'Spectral_r', vmin = inv_min, vmax = inv_max)
        ax[3, i].set_xticks([])
        ax[3, i].set_yticks([])
        plt.colorbar(im3, ax=ax[3, i])

    ax[0, 0].set_ylabel('GP')
    ax[1, 0].set_ylabel('Fixed Mean')
    ax[2, 0].set_ylabel('Fwd Transform')
    ax[3, 0].set_ylabel('Inv Transform')

    fig.suptitle(f"Test data for lengthscale {lengthscale:.2f}")
    plt.savefig(FIGPATH + figname, dpi = 600)
    plt.close()

m = 30
locs = torch.load(FILEPATH + LOCS).numpy()
order = maxmin_approx(locs)
NN = NN_L2(locs[order], m)[:, 1:]


locs = torch.from_numpy(locs[order])
NN = torch.from_numpy(NN)
y = torch.load(FILEPATH + GP_DATA)

scale = compute_scal(locs, NN)

fit_05 = fit_map_mini(y[:100], NN, linear=False, scal=scale, lr=1e-4)
plot_experiment(50, FIGPATH + 'marg_05.png', fit_05, 0.5)

fit_02 = fit_map_mini(y[100:200], NN, linear=False, scal=scale, lr=1e-4)
plot_experiment(150, FIGPATH + 'marg_02.png', fit_02, 0.2)

fit_08 = fit_map_mini(y[200:], NN, linear=False, scal=scale, lr=1e-4)
plot_experiment(250, FIGPATH + 'marg_08.png', fit_08, 0.8)
