import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
from scipy.spatial.distance import cdist


def NN_L2(locs: 'numpy.ndarray', m: 'int') -> 'numpy.ndarray':
    n, d = locs.shape
    NN = - np.ones((n, m + 1), dtype=int)
    mult = 2
    maxVal = min(m * mult + 1, n)
    distM = cdist(locs[:maxVal, :], locs[:maxVal, :])
    odrM = np.argsort(distM)
    for i in range(maxVal):
        NNrow = odrM[i, :]
        NNrow = NNrow[NNrow <= i]
        NNlen = min(NNrow.shape[0], m + 1)
        NN[i, :NNlen] = NNrow[:NNlen]
    queryIdx = np.arange(maxVal, n)
    mSearch = m
    while (queryIdx.size > 0):
        maxIdx = queryIdx.max()
        mSearch = min(maxIdx+1, 2 * mSearch)
        index = NearestNeighbors(n_neighbors=int(mSearch), algorithm='kd_tree').\
            fit(locs[:maxIdx+1, :])
        _, NNsub = index.kneighbors(locs[queryIdx, :])
        lessThanI = NNsub <= queryIdx[:, None]
        numLessThanI = lessThanI.sum(1)
        idxLessThanI = np.nonzero(np.greater_equal(numLessThanI, m + 1))[0]
        for i in idxLessThanI:
            NN[queryIdx[i]] = NNsub[i, lessThanI[i, :]][:m+1]
            if NN[queryIdx[i], 0] != queryIdx[i]:
                try:
                    idx = np.nonzero(NN[queryIdx[i]] == queryIdx[i])[0][0]
                    NN[queryIdx[i], idx] = NN[queryIdx[i], 0]
                    NN[queryIdx[i], 0] = queryIdx[i]
                except IndexError as inst:
                    NN[queryIdx[i], 0] = queryIdx[i]
        queryIdx = np.delete(queryIdx, idxLessThanI, 0)
    if np.any(NN[:, 0] != np.arange(n)):
        warnings.warn("There are very close locations and NN[:, 0] != np.arange(n)\n")
    return NN

