import numpy as np
import faiss
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
        if n < 1e5:
            index = faiss.IndexFlatL2(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(maxIdx+1, 1024))
            index.train(locs[:maxIdx+1, :])
            index.nprobe = min(maxIdx+1, 256)
        index.add(locs[:maxIdx+1, :])
        _, NNsub = index.search(locs[queryIdx, :], int(mSearch))
        lessThanI = NNsub <= queryIdx[:, None]
        numLessThanI = lessThanI.sum(1)
        idxLessThanI = np.nonzero(np.greater_equal(numLessThanI, m + 1))[0]
        for i in idxLessThanI:
            NN[queryIdx[i]] = NNsub[i, lessThanI[i, :]][:m+1]
        queryIdx = np.delete(queryIdx, idxLessThanI, 0)
    return NN

