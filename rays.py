import numpy as numpy
from numba import jit


@jit
def position(origin, direction, distance):
    return origin + direction*distance


def apply(transform, origin, direction):
    return transform(origin), transform(direction)


