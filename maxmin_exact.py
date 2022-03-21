import torch
import maxmin_cpp
import numpy as np

def maxmin_exact(X):
	"""PyTorch implementation of exact max min from GPvecchia:
	https://github.com/katzfuss-group/GPvecchia/blob/master/src/MaxMin.cpp.
	
	X : (Nxd) torch tensor. Does not support batches. 
	
	returns (N,) torch tensor of location in max min ordering. """
	if X.dim() > 2:
		raise Exception("maxmin_exact does not support batch operations.")
	if X.dim() < 2:
		raise Exception("X must be a 2 dimensional tensor.")

	return maxmin_cpp.MaxMincpp(X).type(torch.LongTensor) - 1


def _grouped_maxmin_exact(X):
    N = X.shape[0]

    if N > 250:
        ord1 = _grouped_maxmin_exact(X[0:int(N//2)]).view(-1, 1)
        ord2 = _grouped_maxmin_exact(X[int(N//2):]).view(-1,1) + N // 2
        if N % 2 == 0:
            ord = torch.cat((ord1, ord2), axis = -1).view(-1, 1)
        else:
            ord = torch.cat((
            torch.cat((ord1,ord2[0:-1]), axis = -1).view(-1,1), 
            ord2[-1].view(-1,1)))
        return ord
    else:
        return maxmin_exact(X)

def grouped_maxmin_exact(X):
    ind  = np.arange(0, X.shape[0])
    ind_og = np.arange(0, X.shape[0])
    np.random.shuffle(ind)
    if (X.shape[0] < 10000)&(X.shape[-1] < 500):
        return ind_og[ind[_grouped_maxmin_exact(X[ind]).squeeze()]]
    else:
        return ind
    #return ind