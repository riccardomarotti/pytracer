import numpy as numpy
from numba import jit


@jit
def position(origin, direction, distance):
    return origin + direction*distance
