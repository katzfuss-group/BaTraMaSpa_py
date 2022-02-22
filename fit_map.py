import sys
import torch
from gpytorch.kernels import MaternKernel
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from pyro.distributions import InverseGamma


def nug_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[1]).add(theta[0]))


def scaling_fun(k, theta):
    return torch.sqrt(torch.exp(k.mul(theta[2])))


def sigma_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[4]).add(theta[3]))


def range_fun(theta):
    return torch.exp(theta[5])


def varscale_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[7]).add(theta[6]))


def con_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[9]).add(theta[8]))


def m_threshold(theta, mMax):
    below = scaling_fun(torch.arange(mMax).add(1), theta) < .01
    if below.sum().equal(torch.tensor(0)):
        m = mMax
    else:
        m = torch.argmax(below)
    return torch.maximum(m, torch.tensor(1))


def kernel_fun(X1, theta, sigma, smooth, nuggetMean=None, X2=None):
    N = X1.shape[1]
    if X2 is None:
        X2 = X1
    if nuggetMean is None:
        nuggetMean = 1
    X1s = X1.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    X2s = X2.mul(scaling_fun(torch.arange(1, N + 1).unsqueeze(0), theta))
    lin = X1s @ X2s.t()
    MaternObj = MaternKernel(smooth)._set_lengthscale(range_fun(theta))
    nonlin = MaternObj.forward(X1s, X2s).mult(sigma.pow(2))
    return (lin + nonlin).div(nuggetMean)


def fit_map(data, NNmax, theta, m=None, tuneParm=None, mode='fit', inds=None,
            scal=None):
    if tuneParm is None:
        nugMult = torch.tensor(4.0)
        smooth = torch.tensor(1.5)
        tuneParm = torch.tensor([nugMult, smooth])
    else:
        nugMult = tuneParm[0]
        smooth = tuneParm[1]
    n, N = data.shape
    if m is None:
        m = m_threshold(theta, NNmax.shape[1])
    if inds is None:
        inds = torch.arange(N)
    if scal is None:
        scal = torch.div(torch.tensor(1), torch.arange(N).add(1))
    nHat = inds.shape[0]
    NN = NNmax[:, :m]

    # init some parms
    K = torch.zeros(N, n, n)
    G = torch.zeros(N, n, n)
    GChol = torch.zeros(N, n, n)
    yTilde = torch.zeros(N, n)
    alphaPost = torch.zeros(N)
    betaPost = torch.zeros(N)
    loglik = torch.zeros(N)

    nugMean = nug_fun(torch.arange(N), theta, scal)  # N,
    nugSd = nugMean.mul(nugMult)  # N,
    alpha = nugMean.pow(2).div(nugSd.pow(2)).add(2)  # N,
    beta = nugMean.mul(alpha.sub(1))  # N,

    for i in inds:
        if i == 0:
            G[i, :, :] = torch.eye(n)
        else:
            ncol = torch.minimum(i, m)
            X = data[:, NN[i, :ncol]]  # n X ncol
            K[i, :, :] = kernel_fun(X, theta, sigma_fun(i, theta, scal),
                                    smooth, nugMean[i])  # n X n
            G[i, :, :] = K[i, :, :] + torch.eye(n)  # n X n
    try:
        GChol[inds, :, :] = torch.linalg.cholesky(G[inds, :, :])
    except RuntimeError as inst:
        print(inst)
        if mode == 'fit':
            sys.exit('chol failed')
        else:
            return torch.tensor(float('-inf'))
    yTilde[inds, :] = torch.triangular_solve(data[:, inds].t().unsqueeze(2),
                                             GChol[inds, :, :],
                                             upper=False)[0].squeeze()
    alphaPost[inds] = alpha[inds].add(n / 2)  # N,
    betaPost[inds] = beta[inds] + yTilde[inds, :].square().sum(dim=1).div(2)  # N,
    if mode == 'fit':
        # variable storage has been done through batch operations
        pass
    elif mode == 'intlik':
        # integrated likelihood
        logdet = GChol[inds, :, :].diagonal(dim1=-1, dim2=-2). \
            log().sum(dim=1)  # nHat,
        loglik[inds] = -logdet + alpha[inds].mul(beta[inds].log()) - \
                       alphaPost[inds].mul(betaPost[inds].log()) + \
                       alphaPost[inds].lgamma() - \
                       alpha[inds].lgamma()  # nHat,
    else:
        # profile likelihood
        nuggetHat = betaPost[inds].div(alphaPost[inds].add(1))  # nHat
        fHat = torch.triangular_solve(K[inds, :, :],
                                      GChol[inds, :, :],
                                      upper=False)[0]. \
            bmm(yTilde[inds, :].unsqueeze(2)).squeeze()  # nHat X n
        uniNDist = Normal(loc=fHat, scale=nuggetHat.unsqueeze(1))
        mulNDist = MultivariateNormal(loc=torch.zeros(1, n),
                                      covariance_matrix=K[inds, :, :])
        invGDist = InverseGamma(concentration=alpha[inds], rate=beta[inds])
        loglik[inds] = uniNDist.log_prob(data[:, inds].t()).sum(dim=1) + \
                       mulNDist.log_prob(fHat) + \
                       invGDist.log_prob(nuggetHat)
    if mode == 'fit':
        return GChol, yTilde, nugMean, alphaPost, betaPost, scal, data, \
               data, NN, theta, tuneParm
    else:
        return loglik.sum()