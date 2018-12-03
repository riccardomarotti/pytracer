import numpy as np
from numba import jit


def create(origin, direction):
    return np.array([origin, direction])


def origin(ray):
    return ray[0]


def direction(ray):
    return ray[1]


def position(ray, distance):
    return position_fast(origin(ray), direction(ray), distance)


@jit
def position_fast(origin, direction, distance):
    return origin + direction*distance


def apply(transform, origin, direction):
    return transform(origin), transform(direction)
