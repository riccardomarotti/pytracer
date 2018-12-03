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

    def prepare_computations(self, ray):
        return Computations(self, ray)


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
                min = intersection.t

        return hit_intersection


class Computations:
    def __init__(self, intersection, ray):
        self._t = intersection.t
        self._object = intersection.object
        self._point = ray.position(self._t)
        self._eyev = -ray.direction
        self._normalv = self._object.normal_at(self._point)
        self._inside = self.normalv.dot(self.eyev) < 0
        if self.inside:
            self._normalv = -self.normalv

    @property
    def t(self):
        return self._t

    @property
    def object(self):
        return self._object

    @property
    def point(self):
        return self._point

    @property
    def eyev(self):
        return self._eyev

    @property
    def normalv(self):
        return self._normalv

    @property
    def inside(self):
        return self._inside


@jit(['float32(float32)'])
def hit_fast(ts):
    if len(ts) == 0:
        return np.empty(0)

    min = np.inf

    for t in ts:
        if t >= 0 and t < min:
            min = t

    return np.array([min])
