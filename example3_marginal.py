import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from fit_map import compute_scal, cond_samp, fit_map_mini, TransportMap
from maxmin_approx import maxmin_approx
from NNarray import NN_L2