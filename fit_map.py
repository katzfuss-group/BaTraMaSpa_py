import sys
import torch
from gpytorch.kernels import MaternKernel
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from pyro.distributions import InverseGamma
from torch.nn.parameter import Parameter


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


class TransportMap(torch.nn.Module):
    def __init__(self, thetaInit, linear=False, tuneParm=None):
        super().__init__()
        if tuneParm is None:
            self.nugMult = torch.tensor(4.0)
            self.smooth = torch.tensor(1.5)
        else:
            self.nugMult = tuneParm[0]
            self.smooth = tuneParm[1]
        self.theta = Parameter(thetaInit)
        self.linear = linear

    def forward(self, data, NNmax, mode, m=None, inds=None, scal=None):
        # theta as intermediate var
        if self.linear:
            theta = torch.cat((self.theta,
                               torch.tensor([-float('inf'), .0, .0])))
        else:
            theta = torch.tensor(self.theta)
        # default opt parms
        n, N = data.shape
        if m is None:
            m = m_threshold(theta, NNmax.shape[1])
        if inds is None:
            inds = torch.arange(N)
        if scal is None:
            scal = torch.div(torch.tensor(1), torch.arange(N).add(1))
        NN = NNmax[:, :m]
        # init tmp vars
        K = torch.zeros(N, n, n)
        G = torch.zeros(N, n, n)
        GChol = torch.zeros(N, n, n)
        yTilde = torch.zeros(N, n)
        alphaPost = torch.zeros(N)
        betaPost = torch.zeros(N)
        loglik = torch.zeros(N)
        # Prior vars
        nugMean = nug_fun(torch.arange(N), theta, scal)  # N,
        nugSd = nugMean.mul(self.nugMult)  # N,
        alpha = nugMean.pow(2).div(nugSd.pow(2)).add(2)  # N,
        beta = nugMean.mul(alpha.sub(1))  # N,
        # nll
        for i in inds:
            if i == 0:
                G[i, :, :] = torch.eye(n)
            else:
                ncol = torch.minimum(i, m)
                X = data[:, NN[i, :ncol]]  # n X ncol
                K[i, :, :] = kernel_fun(X, theta, sigma_fun(i, theta, scal),
                                        self.smooth, nugMean[i])  # n X n
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
            tuneParm = torch.tensor([self.nugMult, self.smooth])
            return GChol, yTilde, nugMean, alphaPost, betaPost, scal, data, \
                   data, NN, theta, tuneParm
        else:
            return loglik.sum()


def fit_map_mini(data, NNmax, linear=False, maxIter=1e3, batsz=128,
                 tuneParm=None, lr=1e-3, **kwargs):
    # default initial values
    thetaInit = torch.tensor([data[:, 0].square().mean().log(),
                             .2, -1.0, .0, .0, -1.0])
    if linear:
        thetaInit = thetaInit[0:3]
    transportMap = TransportMap(thetaInit, linear=linear,
                                tuneParm=tuneParm)
    optimizer = torch.optim.SGD(transportMap.parameters(), lr=lr)
    for i in range(maxIter):
        inds = torch.multinomial(torch.ones(data.shape[1]), batsz)
        optimizer.zero_grad()
        loss = transportMap(data, NNmax, 'intlik', inds=inds, **kwargs)
        loss.backward()
        optimizer.step()
        if i % 99 == 0:
            print(f"Loglikelihood {torch.neg(loss)}\n")
    return transportMap(data, NNmax, 'fit')