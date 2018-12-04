import numpy as np
from numba import jit


class Ray:
    def __init__(self, origin, direction):
        self._origin = origin
        self._direction = direction

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    def position(self, distance):
        return position_fast(self.origin, self.direction, distance)

    def transform(self, transform):
        return Ray(transform.dot(self.origin), transform.dot(self.direction))


@jit
def position_fast(origin, direction, distance):
    return origin + direction*distance
