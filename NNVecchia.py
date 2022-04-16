import torch


# Build a NN whose input is (m + 1) X d + m
# First d are locs of the unknown response
# Each subsequent d + 1 are locs and response of a known response
def build_NNVecc(m, d, *args):
    nIn = (m + 1) * d + m
    return build_gen(nIn, 0, 1, *args)


def build_gen(n, pIn, pOut, *args):
    modules = []
    nIn = n + pIn
    for nOut in args:
        modules.append(torch.nn.Linear(nIn, nOut))
        modules.append(torch.nn.ReLU())
        nIn = nOut
    modules.append(torch.nn.Linear(nIn, pOut))
    return torch.nn.Sequential(*modules)


def build_dis(n, *args):
    return build_gen(n, 0, 1, *args)


