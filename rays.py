import numpy as numpy
from numba import jit
import transformations


@jit
def position(origin, direction, distance):
    return origin + direction*distance


def translation(x, y, z):
    T = transformations.translation(x, y, z)
    return lambda origin, direction: (T(origin), T(direction))


def scaling(x, y, z):
    T = transformations.scaling(x, y, z)
    return lambda origin, direction: (T(origin), T(direction))
