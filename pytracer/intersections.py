import numpy as np
from numba import jit


@jit(['float32(float32)'])
def hit(intersections):
    if len(intersections == 0):
        return np.empty(0)

    min = np.inf

    for intersection in intersections:
        if intersection >= 0 and intersection < min:
            min = intersection

    return np.array(min)
