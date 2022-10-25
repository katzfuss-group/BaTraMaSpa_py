from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import torch
from torch.distributions import Normal

from maxmin_approx import maxmin_approx
from NNarray import NN_L2
from fit_map import fit_map_mini, compute_scal, cond_samp, covar_samples


FIGPATH = '../figures/covariate_experiments/'
DATAPATH = '../data/simulations/covariate_experiments/'


def make_uniform_grid(n=40, d=2):
    """Construct a uniform grid of n points in [0, 1]^d."""
    x = np.linspace(0, 1, n)
    grid = np.meshgrid(*[x for _ in range(d)])
    return np.c_[[grid[_].flatten() for _ in range(d)]].T.astype('float32')


def make_kernel(x: float, *args, **kwargs) -> np.ndarray:
    """Construct a kernel matrix for a uniform grid of n points in [0, 1]^d."""
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


def fit_map(x = (0.5, 0.2, 0.8), nsamples = 100, seed = 1, n = 20, d = 2, *args, **kwargs) -> Tuple[dict, dict, dict]:
    initial_params = {'nsamples': nsamples, 'x': x, 'seed': seed, 'n': n, 'd': d}

    if isinstance(x, float):
        x = (x,)
    
    y = construct_gp(nsamples, x, seed, n=n, d=d, *args, **kwargs)
    X = torch.cat([
        torch.ones(nsamples, n**2).mul(_) for _ in x
    ], dim = 0).log().unsqueeze(-1)

    ## If marginal x
    # X = torch.zeros(nsamples, n**2, 1)

    m = 30
    locs = make_uniform_grid(n=n, d=d)
    order = maxmin_approx(locs)
    locs = locs[order]
    nn = NN_L2(locs, m)

    locs = torch.from_numpy(locs)
    nn = torch.from_numpy(nn)[:, 1:]
    scale = compute_scal(locs, nn)

    exp_data = {'locs': locs, 'nn': nn, 'scale': scale, 'order': order}

    tm = fit_map_mini(y, X, nn, scal = scale, linear = False, lr = 8e-7, maxEpoch=25)

    return tm, initial_params, exp_data


def main(sample_index, figname, tm, initial_params, exp_data):
    """Plots results from experiments run in fit_map function."""

    n: int = initial_params['n']
    X: torch.tensor = tm['X_data']
    Y: torch.tensor = tm['Y_data']
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
            fx_sample = covar_samples(tm, 'fx', X_obs = X[sample_index+i])
            fx_min = min(fx_min, fx_sample.min())
            fx_max = max(fx_max, fx_sample.max())
            fx_samples.append(fx_sample)

            fwd_sample = covar_samples(tm, 'trans', Y_obs = Y[sample_index+i], X_obs = X[sample_index+i])
            fwd_min = min(fwd_min, fwd_sample.min())
            fwd_max = max(fwd_max, fwd_sample.max())
            fwd_samples.append(fwd_sample)
            
            inv_sample = covar_samples(tm, 'invtrans', Y_obs = fwd_sample, X_obs = X[sample_index+i])
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

    fig.suptitle(f"Test data for X = log({X[sample_index].mean().exp():.2f})")
    plt.savefig(FIGPATH + figname, dpi = 600)
    plt.close()


if __name__ == "__main__":

    tm, initial_params, exp_data = fit_map()
    n = initial_params['n']
    torch.manual_seed(1)

    with torch.no_grad():
        Y = tm['Y_data']
        X = tm['X_data']
        Z = torch.zeros(Y.shape)
        for t in range(Y.shape[0]):
            Z[t] = covar_samples(tm, 'trans', Y_obs = Y[t], X_obs = X[t])

        torch.save(Z, DATAPATH + 'fwd_transform.pt')

    with torch.no_grad():
        newx = (0.21, 0.3, 0.4, 0.49, 0.51, 0.6, 0.7, 0.79)

        for x in newx:
            X = torch.ones(n**2, 1).mul(x).log()
            Y = torch.empty(25, n**2)
            for j in range(25):
                z = torch.randn(n**2)
                Y[j] = covar_samples(tm, 'invtrans', z, X)

            datum = 0
            fig, ax = plt.subplots(5, 5)
            fig.suptitle('Map samples from X = log({:.2f})'.format(x))
            for row in range(5):
                for col in range(5):
                    ax[row, col].imshow(Y[datum].reshape(n, n), cmap = 'Spectral_r')
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])
                    datum += 1
            plt.savefig(FIGPATH + f"new_sample_{x}.png", dpi = 600)
            plt.close()

    ## Uncomment to reproduce simulations reproducing the original GPs
    # sample_indices = (50, 150, 250)
    # fignames = ('log05.png', 'log02.png', 'log08.png')

    # for sample_index, figname in zip(sample_indices, fignames):
    #     main(sample_index, figname, tm, initial_params, exp_data)

    ## Uncomment to reproduce simulations reproducing the original GPs marginally
    ## Uncomment the X modification in main() if fitting marginally
    # xvals = (0.5, 0.2, 0.8)
    # fignames = ('nox_log05.png', 'nox_log02.png', 'nox_log08.png')
    # for xval, figname in zip(xvals, fignames):
    #     tm, initial_params, exp_data = fit_map(x = xval)
    #     sample_index = 50
    #     main(sample_index, figname, tm, initial_params, exp_data)