import numpy as np
from numba import jit


@jit
def point(x, y, z):
    return np.array([x, y, z, 1.0])


@jit
def vector(x, y, z):
    return np.array([x, y, z, 0.0])
