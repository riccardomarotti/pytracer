import numpy as np
import math
from pytracer.tuples import point, vector, normalize
from pytracer.transformations import identity_matrix
from pytracer.transformations import invert
from pytracer.transformations import transpose
import pytracer.materials
import pytracer.rays as rays
from numba import jit


def sphere(transformation=identity_matrix, material=None):
    if material is None:
        material = pytracer.materials.material()

    return [transformation, material]


def transformation(sphere):
    return sphere[0]


def material(sphere):
    return sphere[1]


def normal_at(p, transform=identity_matrix):
    transform = invert(transform)
    object_point = transform(p)
    object_normal = object_point - point(0, 0, 0)
    world_normal = transpose(transform)(object_normal)
    world_normal[3] = 0
    return normalize(world_normal)


def intersect(origin, direction, transformation=identity_matrix):
    transformation = invert(transformation)
    origin, direction = rays.apply(transformation, origin, direction)
    return intersect_fast(origin, direction)


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
