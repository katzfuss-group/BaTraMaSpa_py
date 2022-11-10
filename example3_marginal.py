import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal

from fit_map import compute_scal, cond_samp, fit_map_mini, TransportMap
from maxmin_approx import maxmin_approx
from NNarray import NN_L2

# Fig path and files to read from a higher directory
# gp data and locs generated using example3.construct_gp
# and example3.make_uniform_grid. Files saved via torch.save.
FIGPATH = '../figures/covariate_experiments/'
FILEPATH = '../data/simulations/covariate_experiments/'
GP_DATA = 'gp_050208.npy'
LOCS = 'locs.npy'

XVALS = (0.5, 0.2, 0.8)

def make_uniform_grid(n=40, d=2):
    """Construct a uniform grid of n points in [0, 1]^d."""
    x = np.linspace(0, 1, n)
    grid = np.meshgrid(*[x for _ in range(d)])
    return np.c_[[grid[_].flatten() for _ in range(d)]].T.astype('float32')


def make_kernel(x: float, *args, **kwargs) -> np.ndarray:
    """Construct a kernel matrix for a uniform grid of n points in [0, 1]^d."""
    from scipy.spatial.distance import cdist
    locs = make_uniform_grid(*args, **kwargs)
    dist = cdist(locs, locs)
    p = dist.shape[0] // 2
    kernel = np.exp(-dist / x)

    return kernel


def sample_gp(samples, *args, **kwargs):
    kernel = torch.from_numpy(make_kernel(*args, **kwargs))
    samples = torch.tensor((samples, kernel.shape[0]))
    z = Normal(0.0, 1.0).sample(samples)
    C = torch.linalg.cholesky(kernel).float()
    y = C.matmul(z.T)
    return y.T


def construct_gp(
    samples: int = 100, x: tuple = (0.5, 0.2, 0.8), seed: int = 1, *args, **kwargs
) -> torch.Tensor:
    torch.manual_seed(seed)

    # want two iterables with equal samples for each x
    if isinstance(x, float):
        x = (x,)
    samples = [samples for _ in range(len(x))]

    assert len(samples) == len(x)
    n = kwargs["n"] if "n" in kwargs.keys() else 40

    y = torch.empty((len(samples), samples[0], n**2))

    for k in range(len(samples)):
        y[k] = sample_gp(samples[k], x=x[k], *args, **kwargs)

    return y.reshape(-1, n**2)


def plot_experiment(sample_index, figname, tm, lengthscale):
    """
    Modification of example3.main from this repo. Plots experiments in
    using a marginal transport map only.
    """

    scale: torch.tensor = tm['scal']

    with torch.no_grad():
        fx_samples = []
        fwd_samples = []
        inv_samples = []

        fx_min  = fx_max  = 0
        fwd_min = fwd_max = 0
        inv_min = inv_max = 0

        for i in range(6):
            fx_sample = cond_samp(tm, 'fx')
            fx_min = min(fx_min, fx_sample.min())
            fx_max = max(fx_max, fx_sample.max())
            fx_samples.append(fx_sample)

            fwd_sample = cond_samp(tm, 'trans', obs = Y[sample_index + i])
            fwd_min = min(fwd_min, fwd_sample.min())
            fwd_max = max(fwd_max, fwd_sample.max())
            fwd_samples.append(fwd_sample)
            
            inv_sample = cond_samp(tm, 'invtrans', obs = fwd_sample)
            inv_min = min(inv_min, inv_sample.min())
            inv_max = max(inv_max, inv_sample.max())
            inv_samples.append(inv_sample)

        Z = torch.empty(Y.shape)
        for _ in range(Z.shape[0]):
            Z[_] = cond_samp(tm, 'trans', obs = Y[_])
        
    torch.save(Z, os.path.join(FILEPATH, f'margZ_{lengthscale}.pt'))

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
n = 20
Y = construct_gp(samples = 100, n = n)

scale = compute_scal(locs, NN)

fit_05 = fit_map_mini(Y[:100], NN, linear=False, scal=scale, lr=1e-5)
plot_experiment(50, 'marg_05.png', fit_05, 0.5)

fit_02 = fit_map_mini(Y[100:200], NN, linear=False, scal=scale, lr=1e-5)
plot_experiment(50, 'marg_02.png', fit_02, 0.2)

fit_08 = fit_map_mini(Y[200:], NN, linear=False, scal=scale, lr=1e-5)
plot_experiment(50, 'marg_08.png', fit_08, 0.8)
