import numpy as np
from numba import jit


class Intersection:
    def __init__(self, t, object):
        self._object = object
        self._t = t

    @property
    def object(self):
        return self._object

    @property
    def t(self):
        return self._t


class Intersections:
    def __init__(self, *intersections):
        self._intersections = intersections

    def __getitem__(self, key):
        return self._intersections[key]

    def __len__(self):
        return len(self._intersections)

    def hit(self):
        hit_intersection = None
        if len(self._intersections) == 0:
            return hit_intersection

        min = np.inf

        for intersection in self._intersections:
            if intersection.t >= 0 and intersection.t < min:
                hit_intersection = intersection

        return hit_intersection


@jit(['float32(float32)'])
def hit_fast(ts):
    if len(ts) == 0:
        return np.empty(0)

    min = np.inf

    for t in ts:
        if t >= 0 and t < min:
            min = t

    return np.array([min])
