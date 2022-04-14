import torch


# Build a NN whose input is (m + 1) X d + m
# First d are locs of the unknown response
# Each subsequent d + 1 are locs and response of a known response
def build_NNVecc(m, d, *args):
    modules = []
    nIn = (m + 1) * d + m
    for nOut in args:
        modules.append(torch.nn.Linear(nIn, nOut))
        modules.append(torch.nn.ReLU())
        nIn = nOut
    nOut = 1
    modules.append(torch.nn.Linear(nIn, nOut))
    return torch.nn.Sequential(*modules)


