import numpy as np
from numba import jit


def create(object, t):
    return np.array(object, t)


def t(intersection):
    return intersection[1]


def object(intersection):
    return intersection[0]


@jit(['float32(float32)'])
def hit(intersections):
    if len(intersections) == 0:
        return np.empty(0)

    min = np.inf

    for intersection in intersections:
        if intersection >= 0 and intersection < min:
            min = intersection

    return np.array([min])
