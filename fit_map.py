import sys
import torch
from gpytorch.kernels import MaternKernel
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from pyro.distributions import InverseGamma
from torch.nn.parameter import Parameter
from torch.distributions.studentT import StudentT

# region: help-functions
def nug_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[1]).add(theta[0]))


def scaling_fun(k, theta):
    return torch.sqrt(torch.exp(k.mul(theta[2])))


def scaling_x(scal, theta, index0, index1):
    return scal.log().mul(theta[index1]).add(theta[index0]).exp()

def linear_scaling_x(scal, theta, index0, index1):
    return scaling_x(scal, theta, index0.sub(3), index1.sub(3))


def sigma_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[4]).add(theta[3]))


def range_fun(theta):
    return torch.exp(theta[5])


# TODO: Confirm if this can be removed from the project.
def varscale_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[7]).add(theta[6]))


# TODO: Confirm if this code can be removed from the project.
def con_fun(i, theta, scales):
    return torch.exp(torch.log(scales[i]).mul(theta[9]).add(theta[8]))


def m_threshold(theta, mMax):
    below = scaling_fun(torch.arange(mMax).add(1), theta) < 0.01
    # Switch these cases around.  The test is equivalent to below.all() == False.
    # A more clear statement will be easier to understand (and maintain).
    if below.sum().equal(torch.tensor(0)):
        m = torch.tensor(mMax)
    else:
        m = torch.argmax(below.type(torch.DoubleTensor))
    return torch.maximum(m, torch.tensor(1))


# Peak where this is called to understand what the input is.
# Calls once in TransportMap.forward and twice in cond_sample.
def kernel_fun(Y1, X1, theta, sigma, smooth, scal, nuggetMean=None, Y2=None, X2=None, linear=False):
    # Y1 assumed n x N
    N = Y1.shape[1]

    # X1 assumed (n x N x p) with subindexing to (n x p) at the ith location
    # (i = 0, ..., N-1).  Must construct indexing for theta_{x0} and theta_{x1}.
    # The 6 is hard coded based on an original prior.  In the future this should
    # all be handled by class attributes rather than hard coding.
    n = X1.shape[0]
    p = X1.shape[-1]

    index0 = torch.arange(p).add(6)
    index1 = torch.arange(p).add(6 + p)

    # Now must force the scale to have the same shape as X1.  The scale at the
    # ith location is the same for all n and p, and is assumed to be a scalar
    # tensor. The scale is then expanded to have the same shape as X1.
    if not isinstance(scal, torch.Tensor):
        scal = torch.tensor(scal)
    scal = scal.repeat(n, p)

    # Fill in other locations if necessary.
    if Y2 is None:
        Y2 = Y1
    if X2 is None:
        X2 = X1
    if nuggetMean is None:
        nuggetMean = 1

    # Use a common index for the set of locations to reference in Y.  This works
    # because the kernel computes distances to each point.  A different index is
    # needed in the x case because it is not based on nearest neighbors.
    indices = torch.arange(N).add(1).unsqueeze(0)
    Y1s = Y1.mul(scaling_fun(indices, theta))
    Y2s = Y2.mul(scaling_fun(indices, theta))

    # This version of the map should handle a case when X is identically zero,
    # meaning we've supplied no covariates. In this case, we should simply update
    # X1s and X2s as zero to pass into the kernel.
    if X1.eq(0).all():
        X1s = torch.zeros_like(X1)
        X2s = torch.zeros_like(X2)
    else:
        if linear:
            X1s = X1.mul(scaling_x(scal, theta, index0, index1))
            X2s = X2.mul(scaling_x(scal, theta, index0, index1))
        else:
            X1s = X1.mul(linear_scaling_x(scal, theta, index0, index1))
            X2s = X2.mul(linear_scaling_x(scal, theta, index0, index1))
    
    # Now concatenate Y and X components for use in the full kernel.
    W1s = torch.cat((Y1s, X1s), 1)
    W2s = torch.cat((Y2s, X2s), 1)
    lin = W1s @ W2s.t()  # n x n
    MaternObj = MaternKernel(smooth.item())
    MaternObj._set_lengthscale(1.0)
    lenScal = range_fun(theta) * smooth.mul(2).sqrt()
    nonlin = MaternObj.forward(W1s.div(lenScal), W2s.div(lenScal)).mul(sigma.pow(2))
    return (lin + nonlin).div(nuggetMean)


# endregion: help-functions


class TransportMap(torch.nn.Module):
    # Initialization may need to be modified for covariates to include a longer
    # hyperprior theta.
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

    def forward(self, Y_data, X_data, NNmax, mode, m=None, inds=None, scal=None):
        # theta as intermediate var
        if self.linear:
            theta = torch.cat((self.theta, torch.tensor([-float("inf"), 0.0, 0.0])))
        else:
            theta = self.theta
        # Get dimensions from X_data.  Assume X is (n, N, p) and Y is (n, N).
        # Validate assumptions by comparing the shapes of X and Y data.
        n, N, p = X_data.shape

        assert Y_data.shape == (n, N)

        if m is None:
            m = m_threshold(theta, NNmax.shape[1])
        if inds is None:
            inds = torch.arange(N)

        Nhat = inds.shape[0]

        # This must be included in the kernel function.
        if scal is None:
            scal = torch.div(torch.tensor(1), torch.arange(N).add(1))  # N

        NN = NNmax[:, :m]
        # init tmp vars
        K = torch.zeros(Nhat, n, n)
        G = torch.zeros(Nhat, n, n)
        loglik = torch.zeros(Nhat)

        # Prior vars
        nugMean = torch.relu(nug_fun(inds, theta, scal).sub(1e-5)).add(1e-5)  # Nhat,
        nugSd = nugMean.mul(self.nugMult)  # Nhat,
        alpha = nugMean.pow(2).div(nugSd.pow(2)).add(2)  # Nhat,
        beta = nugMean.mul(alpha.sub(1))  # Nhat,
        # nll
        for i in range(Nhat):
            if inds[i] == 0:
                G[i, :, :] = torch.eye(n)
            else:
                # Pick the nearest neighbors of the ith point to use in kernel.
                # Then concatenate a vector of information at point i to the
                # nearest neighbors for use in a kernel.

                # Operation to stack the vectors will be something like
                # torch.cat((Y_NN, X_i), dim = 1).

                # Either kernel_fun should be modified to take two arguments or
                # we should construct a vector W here. Let's go for the kernel_fun
                # modification for now.
                ncol = torch.minimum(inds[i], m)
                Y = Y_data[:, NN[inds[i], :ncol]]  # n X ncol
                X = X_data[:, i]
                K[i, :, :] = kernel_fun(
                    Y,
                    X,
                    theta,
                    sigma_fun(inds[i], theta, scal),
                    self.smooth,
                    scal[i],
                    nugMean[i],
                    linear = self.linear
                )  # n X n
                G[i, :, :] = K[i, :, :] + torch.eye(n)  # n X n
        try:
            GChol = torch.linalg.cholesky(G)
        except RuntimeError as inst:
            print(inst)
            if mode == "fit":
                sys.exit("chol failed")
            else:
                return torch.tensor(float("-inf"))
        # TODO: The triangular_solve function is depricated and should be replaced by
        # the commented code below once an initial test of covariates has been
        # completed. The implementation below should be equivalent to the original
        # implementation.
        yTilde = torch.triangular_solve(
            Y_data[:, inds].t().unsqueeze(2), GChol, upper=False
        )[
            0
        ].squeeze()  # Nhat X n
        # yTilde = torch.linalg.solve_triangular(
        #     GChol, Y_data[:, inds].t().unsqueeze(2), upper=False
        # ).squeeze()

        # TODO: confirm if we need to return an xtilde as part of the output?
        alphaPost = alpha.add(n / 2)  # Nhat,
        betaPost = beta + yTilde.square().sum(dim=1).div(2)  # Nhat,
        if mode == "fit":
            # variable storage has been done through batch operations
            pass
        elif mode == "intlik":
            # integrated likelihood
            logdet = GChol.diagonal(dim1=-1, dim2=-2).log().sum(dim=1)  # nHat,
            loglik = (
                -logdet
                + alpha.mul(beta.log())
                - alphaPost.mul(betaPost.log())
                + alphaPost.lgamma()
                - alpha.lgamma()
            )  # nHat,
        else:
            # profile likelihood
            nuggetHat = betaPost.div(alphaPost.add(1))  # nHat
            fHat = (
                torch.triangular_solve(K, GChol, upper=False)[0]
                .bmm(yTilde.unsqueeze(2))
                .squeeze()
            )  # nHat X n
            uniNDist = Normal(loc=fHat, scale=nuggetHat.unsqueeze(1))
            mulNDist = MultivariateNormal(loc=torch.zeros(1, n), covariance_matrix=K)
            invGDist = InverseGamma(concentration=alpha, rate=beta)
            loglik = (
                uniNDist.log_prob(Y_data[:, inds].t()).sum(dim=1)
                + mulNDist.log_prob(fHat)
                + invGDist.log_prob(nuggetHat)
            )
        if mode == "fit":
            tuneParm = torch.tensor([self.nugMult, self.smooth])
            return {
                "Chol": GChol,
                "yTilde": yTilde,
                "nugMean": nugMean,
                "alphaPost": alphaPost,
                "betaPost": betaPost,
                "scal": scal,
                "Y_data": Y_data,
                "X_data": X_data,
                "NN": NN,
                "theta": theta,
                "tuneParm": tuneParm,
            }
        else:
            return loglik.sum().neg()


# This must be modified on account of covariates
# TODO: Investigate making this an instance method of TransportMap
def fit_map_mini(
    Y_data,
    X_data,
    NNmax,
    scal=None,
    linear=False,
    maxEpoch=10,
    batch_size=128,
    tuneParm=None,
    lr=1e-5,
    Y_dataTest=None,
    X_dataTest=None,
    NNmaxTest=None,
    scalTest=None,
    **kwargs
):
    # Assume X_data.shape = [p, n, N] or [n, N] in the p=1 case
    p = 1 if len(X_data.shape) == 2 else X_data.shape[0]
    # default initial values
    thetaInit = torch.tensor(
        # TODO:
        # Revisit this instantiation. How does it work if we use the linear flag?
        # In the linear case we concatenate theta with [-inf, 0, 0], so the
        # input theta should be fine.
        # This should be deparsed into two separate constructors
        # rather than handled as control flow.
        [
            Y_data[:, 0].square().mean().log(), 0.2, #\theta_sigma hypers
            -1.0, 0.0, #\theta_d hypers
            0.0, # range param hyper
            -1.0, # nugget hyper
            X_data.mean(0).mean(0).log(), # x0 hypers assumed to act like locations
            X_data.mean(0).std(0).log() # x1 hypers assumed to act like scales
        ]
    )
    if linear:
        # Must keep the x parameters when subsetting for a linear fit.
        # Temporarily store parameters in a list, then replace it
        _theta_init_size = thetaInit.shape[0]
        _linear_subset = [0, 1, 2] + list(range(6, _theta_init_size))
        thetaInit = thetaInit[_linear_subset]

        # Destroy these variables after recreating the linear theta
        del (_theta_init_size, _linear_subset)

    transportMap = TransportMap(thetaInit, linear=linear, tuneParm=tuneParm)
    optimizer = torch.optim.SGD(transportMap.parameters(), lr=lr, momentum=0.9)
    if Y_dataTest is None:
        Y_dataTest = Y_data[:, : min(Y_data.shape[1], 5000)]
        NNmaxTest = NNmax[: min(Y_data.shape[1], 5000), :]
        if scal is not None:
            scalTest = scal[: min(Y_data.shape[1], 5000)]

    if X_dataTest is None:
        X_dataTest = X_data[:, : min(X_data.shape[1], 5000), :]

    # optimizer = torch.optim.Adam(transportMap.parameters(), lr=lr)
    epochIter = int(Y_data.shape[1] / batch_size)
    for i in range(maxEpoch):
        for j in range(epochIter):
            inds = torch.multinomial(torch.ones(Y_data.shape[1]), batch_size)
            optimizer.zero_grad()
            try:
                loss = transportMap(
                    Y_data, X_data, NNmax, "intlik", inds=inds, scal=scal, **kwargs
                )
                loss.backward()
            except RuntimeError as inst:
                print("Warning: the current optimization iteration failed")
                print(inst)
                continue
            optimizer.step()
        print("Epoch ", i + 1, "\n")
        for name, parm in transportMap.named_parameters():
            print(name, ": ", parm.data)
        if i == 0:
            with torch.no_grad():
                scrPrev = transportMap(Y_dataTest, X_dataTest, NNmaxTest, "intlik", scal=scalTest)
                print("Current test score is ", scrPrev, "\n")
        else:
            with torch.no_grad():
                scrCurr = transportMap(Y_dataTest, NNmaxTest, "intlik", scal=scalTest)
                print("Current test score is ", scrCurr, "\n")
            if scrCurr > scrPrev:
                break
            scrPrev = scrCurr
    with torch.no_grad():
        return transportMap(Y_data, X_data, NNmax, "fit", scal=scal, **kwargs)


# Modify on account of covariates
# (must sample the length scale parameters)
# TODO: Investigate making this an instance method of TransportMap
def cond_samp(fit, mode, Y_obs=None, X_obs = None, xFix=torch.tensor([]), indLast=None):
    Y_data = fit["Y_data"]
    X_data = fit["X_data"]
    NN = fit["NN"]
    theta = fit["theta"]
    scal = fit["scal"]
    nugMult = fit["tuneParm"][0]
    smooth = fit["tuneParm"][1]
    nugMean = fit["nugMean"]
    chol = fit["Chol"]
    yTilde = fit["yTilde"]
    betaPost = fit["betaPost"]
    alphaPost = fit["alphaPost"]

    n, N, p = X_data.shape
    assert Y_data.shape == (n, N)

    m = NN.shape[1]
    if indLast is None:
        indLast = N

    # loop over variables/locations
    y_new = scr = torch.cat((xFix, torch.zeros(N - xFix.size(0))))
    for i in range(xFix.size(0), indLast + 1):
        # predictive distribution for current sample
        if i == 0:
            cStar = torch.zeros(n)
            prVar = torch.tensor(0.0)
        else:
            ncol = min(i, m)
            Y = Y_data[:, NN[i, :ncol]]
            X = X_data[:, i]
            if mode in ["score", "trans", "scorepm"]:
                Y_pred = Y_obs[NN[i, :ncol]].unsqueeze(0)
                X_pred = X_obs
            else:
                Y_pred = y_new[NN[i, :ncol]].unsqueeze(0)
                X_pred = X
            cStar = kernel_fun(
                Y_pred,
                X_pred,
                theta,
                sigma_fun(i, theta, scal),
                smooth,
                scal,
                nugMean[i],
                Y2=Y,
                X2=X,
                linear = fit["linear"]
            ).squeeze()
            prVar = kernel_fun(
                Y_pred,
                X_pred,
                theta,
                sigma_fun(i, theta, scal),
                smooth,
                scal,
                nugMean[i],
                linear = fit["linear"]
            ).squeeze()
        # TODO: Copy the updated linalg.solve_triangular solution from above
        # to fix FutureWarnings about triangular_solve.
        cChol = torch.triangular_solve(cStar.unsqueeze(1), chol[i, :, :], upper=False)[
            0
        ].squeeze()
        meanPred = yTilde[i, :].mul(cChol).sum()
        varPredNoNug = prVar - cChol.square().sum()
        # evaluate score or sample
        if mode == "score":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            STDist = StudentT(2 * alphaPost[i])
            scr[i] = (
                STDist.log_prob((obs[i] - meanPred) / initVar.sqrt())
                - 0.5 * initVar.log()
            )
        elif mode == "scorepm":
            nugget = betaPost[i] / alphaPost[i].sub(1)
            uniNDist = Normal(loc=meanPred, scale=nugget.sqrt())
            scr[i] = uniNDist.log_prob(obs[i])
        elif mode == "fx":
            y_new[i] = meanPred
        elif mode == "freq":
            nugget = betaPost[i] / alphaPost[i].add(1)
            uniNDist = Normal(loc=meanPred, scale=nugget.sqrt())
            y_new[i] = uniNDist.sample()
        elif mode == "bayes":
            invGDist = InverseGamma(concentration=alphaPost[i], rate=betaPost[i])
            nugget = invGDist.sample()
            uniNDist = Normal(loc=meanPred, scale=nugget.mul(1 + varPredNoNug).sqrt())
            y_new[i] = uniNDist.sample()
        elif mode == "trans":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            xStand = (obs[i] - meanPred) / initVar.sqrt()
            STDist = StudentT(2 * alphaPost[i])
            uniNDist = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
            y_new[i] = uniNDist.icdf(STDist.cdf(xStand))
        elif mode == "invtrans":
            initVar = betaPost[i] / alphaPost[i] * (1 + varPredNoNug)
            STDist = StudentT(2 * alphaPost[i])
            uniNDist = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
            y_new[i] = meanPred + STDist.icdf(uniNDist.cdf(obs[i])) * initVar.sqrt()
    if mode in ["score", "scorepm"]:
        return scr.sum()
    else:
        return y_new


# locsOdr: each row is one location
# NN: each row represents one location
# TODO: Investigate moving this to a preprocessing module
def compute_scal(locsOdr, NN):
    N = locsOdr.shape[0]
    scal = (locsOdr[1:, :] - locsOdr[NN[1:, 0], :]).square().sum(1).sqrt()
    scal = torch.cat((scal[0].square().div(scal[4]).unsqueeze(0), scal))
    scal = scal.div(scal[0])
    return scal
