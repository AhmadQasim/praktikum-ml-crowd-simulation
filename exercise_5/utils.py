import numpy as np


def mean_squared_error(x, x_hat):
    return np.square(x - x_hat).sum()
