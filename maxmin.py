import numpy as np
import numpy.ma as ma
from sklearn.neighbors import NearestNeighbors

def order_maxmin(x):
    '''
    Approximate maxmin ordering (https://arxiv.org/abs/1609.05372).
    Python version of middle out ordering from https://rdrr.io/cran/GPvecchia/.
    Taken from https://github.com/katzfuss-group/bayesOpt/blob/main/vecchiaBayesOpt/pyvecch/utils.py
    Inputs
    x : locs information
    returns maxmin order
    '''
    n = x.shape[0]
    m = int(np.round(np.sqrt(n)))
    nbrs_model = NearestNeighbors(n_neighbors=m, algorithm='ball_tree').fit(x)
    _, NNall = nbrs_model.kneighbors(x)
    ind = np.arange(n)
    np.random.shuffle(ind)
    index_in_position = np.ones(shape = (3 * n,), dtype = int)
    index_in_position = ma.array(index_in_position, dtype = int)
    index_in_position[0:] = ma.masked
    index_in_position[0:n] = ind
    position_of_index = np.argsort(index_in_position[0:n])

    curlen = n
    nmoved = 0

    for j in range(1, (2 * n)):
        nneigh = int(np.round(np.min([m, n/(j-nmoved+1)])))
        indy = index_in_position[j:(j+1)].compressed()
        nbrs = NNall[indy,0:nneigh]
        if(np.size(nbrs)>0):
            if(np.min(position_of_index[nbrs]) < j):
                nmoved = nmoved+1
                curlen = curlen + 1
                position_of_index[ indy ] = curlen
                index_in_position[curlen] = index_in_position[j]
                index_in_position[j] = ma.masked
    ord = index_in_position.compressed()
    return(ord)