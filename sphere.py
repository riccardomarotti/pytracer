import numpy as np
import math
from tuples import point, vector
from transformations import identity
from transformations import invert
import rays
from numba import jit


@jit
def intersect(origin, direction, transformation=identity):
    transformation = invert(transformation)
    origin, direction = rays.apply(transformation, origin, direction)
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
