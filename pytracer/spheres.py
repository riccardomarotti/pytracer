import numpy as np
import math
from pytracer.tuples import point, vector, normalize
from pytracer.transformations import identity_matrix
from pytracer.transformations import invert
from pytracer.transformations import transpose
from pytracer.materials import Material

from numba import jit


class Sphere:
    def __init__(self, transformation=identity_matrix, material=None):
        if material is None:
            self._material = Material()
        else:
            self._material = material

        self._transformation = transformation

    @property
    def transformation(self):
        return self._transformation

    @property
    def material(self):
        return self._material

    def normal_at(self, p):
        transformation = invert(self.transformation)
        object_point = transformation(p)
        object_normal = object_point - point(0, 0, 0)
        world_normal = transpose(transformation)(object_normal)
        world_normal[3] = 0
        return normalize(world_normal)

    def intersect(self, ray):
        transformation = invert(self.transformation)
        transformed_ray = ray.transform(transformation)
        return intersect_fast(transformed_ray.origin, transformed_ray.direction)


@jit
def intersect_fast(origin, direction):
    sphere_to_ray = origin - point(0, 0, 0)

    a = direction.dot(direction)
    b = 2*direction.dot(sphere_to_ray)
    c = sphere_to_ray.dot(sphere_to_ray) - 1

    delta = b**2 - 4*a*c

    if delta < 0:
        return np.empty((0))

    t1 = (-b - math.sqrt(delta)) / (2*a)
    t2 = (-b + math.sqrt(delta)) / (2*a)

    return np.array([t1, t2])
