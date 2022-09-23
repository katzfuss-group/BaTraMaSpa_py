import warnings
from numpy import ndarray, any, argsort, arange, delete, greater_equal, int32, int64, nonzero, ones
from scipy.spatial.distance import cdist

import faiss


def NN_L2(locs: ndarray, m: int) -> ndarray:
    n, d = locs.shape
    NN = -ones((n, m + 1), dtype=int)
    mult = 2
    maxVal = min(m * mult + 1, n)
    distM = cdist(locs[:maxVal, :], locs[:maxVal, :])
    odrM = argsort(distM)
    for i in range(maxVal):
        NNrow = odrM[i, :]
        NNrow = NNrow[NNrow <= i]
        NNlen = min(NNrow.shape[0], m + 1)
        NN[i, :NNlen] = NNrow[:NNlen]
    queryIdx = arange(maxVal, n)
    mSearch = m
    while queryIdx.size > 0:
        maxIdx = queryIdx.max()
        mSearch = min(maxIdx + 1, 2 * mSearch)
        if n < 1e5:
            index = faiss.IndexFlatL2(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, min(maxIdx + 1, 1024))
            index.train(locs[: maxIdx + 1, :])
            index.nprobe = min(maxIdx + 1, 256)
        index.add(locs[: maxIdx + 1, :])
        _, NNsub = index.search(locs[queryIdx, :], int(mSearch))
        lessThanI = NNsub <= queryIdx[:, None]
        numLessThanI = lessThanI.sum(1)
        idxLessThanI = nonzero(greater_equal(numLessThanI, m + 1))[0]
        for i in idxLessThanI:
            NN[queryIdx[i]] = NNsub[i, lessThanI[i, :]][: m + 1]
            if NN[queryIdx[i], 0] != queryIdx[i]:
                try:
                    idx = nonzero(NN[queryIdx[i]] == queryIdx[i])[0][0]
                    NN[queryIdx[i], idx] = NN[queryIdx[i], 0]
                    NN[queryIdx[i], 0] = queryIdx[i]
                except IndexError as inst:
                    NN[queryIdx[i], 0] = queryIdx[i]
        queryIdx = delete(queryIdx, idxLessThanI, 0)
    if any(NN[:, 0] != arange(n)):
        warnings.warn("There are very close locations and NN[:, 0] != np.arange(n)\n")
    return NN.astype(int64)
