import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.spatial.distance import cdist
import torch

__all__ = [
    "make_uniform_grid",
    "make_kernel",
    "sample_gp",
    "construct_gp",
]

def make_uniform_grid(n: int = 40, d: int = 2):
    """
    Construct a uniform grid of n points in [0, 1]^d.

    Parameters:
    -----------
    n: int
        Number of points per dimension (assumed square grid)

    d: int
        Number of dimensions
    
    Returns:
    --------
    grid: np.ndarray(dtype=float32)
        Array of shape (n^d, d) containing the grid points
    """
    x = np.linspace(0, 1, n)
    grid = np.meshgrid(*[x for _ in range(d)])
    return np.c_[[grid[_].flatten() for _ in range(d)]].T.astype('float32')


def make_kernel(length_scale: float, **kwargs: dict) -> np.ndarray:
    """
    Construct an exponential kernel with lengthscale length_scale. The kernel is
    constructed on a grid built using make_uniform_grid.

    Parameters:
    -----------
    length_scale: float
        Length scale of an exponential kernel
    
    kwargs: dict
        Keyword arguments to pass to make_uniform_grid

    Returns:
    --------
    kernel: np.ndarray(dtype=float32)
    """

    locs = make_uniform_grid(**kwargs)
    dist = cdist(locs, locs)
    kernel = np.exp(-dist / length_scale, dtype='float32')

    return kernel


def sample_gp(samples: int, **kwargs) -> np.ndarray:
    """
    Constructs a Gaussian process with a given kernel and samples from it.
    Keyword arguments are passed to make_kernel (and in turn to make_uniform_grid).
    """
    kernel = make_kernel(**kwargs)

    z = norm(0.0, 1.0).rvs((kernel.shape[0], samples))
    C = np.linalg.cholesky(kernel)
    y = C @ z

    return y.T


def construct_gp(
    nsamples: int = 100, length_scale: tuple = (0.5, 0.2, 0.8), seed: int = None, **kwargs
) -> np.ndarray:
    """
    Helper to construct nsamples copies of a Gaussian process with a given kernel.
    
    Parameters:
    -----------
    nsamples: int
        Number of replications to generate for each x

    x: tuple
        Tuple of lengthscale parameters to use for the kernel

    seed: int
        Random seed to use for reproducibility
    
    kwargs: dict
        Keyword arguments to pass to make_kernel (and in turn to make_uniform_grid)

    Returns:
    --------
    gaussian_process: np.ndarray(dtype=float32)
        Array of shape (len(length_scale), nsamples, n**2) containing the samples, where
        n^2 is derived from the keyword arguments passed to make_uniform_grid
        representing a unit cube in d dimensions (=2 by default).
    """
    np.random.seed(seed)
    
    # want two iterables with equal samples/length for each x
    if isinstance(length_scale, float):
        length_scale = (length_scale,)
    samples = [nsamples for _ in range(len(length_scale))]

    assert len(samples) == len(length_scale)
    n = kwargs["n"] if "n" in kwargs.keys() else 40

    y = np.empty((len(length_scale), nsamples, n**2))

    for k in range(len(length_scale)):
        y[k] = sample_gp(samples[k], length_scale=length_scale[k], **kwargs)

    return y
