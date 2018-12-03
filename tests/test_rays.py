from pytracer.tuples import point, vector
import pytracer.transformations as transformations
from pytracer.rays import Ray
import numpy as np


def test_computing_a_point_from_a_distance():
    origin = point(2, 3, 4)
    direction = vector(1, 0, 0)
    r = Ray(origin, direction)

    assert((point(2, 3, 4) == r.position(0)).all())
    assert((point(3, 3, 4) == r.position(1)).all())
    assert((point(1, 3, 4) == r.position(-1)).all())
    assert((point(4.5, 3, 4) == r.position(2.5)).all())


def test_translating_a_ray():
    origin = point(1, 2, 3)
    direction = vector(0, 1, 0)
    r = Ray(origin, direction)

    m = transformations.translation(3, 4, 5)
    translated_ray = r.transform(m)

    assert(np.allclose(point(4, 6, 8), translated_ray.origin))
    assert(np.allclose(vector(0, 1, 0), translated_ray.direction))


def test_scaling_a_ray():
    origin = point(1, 2, 3)
    direction = vector(0, 1, 0)
    r = Ray(origin, direction)

    m = transformations.scaling(2, 3, 4)
    scaled_ray = r.transform(m)

    assert(np.allclose(point(2, 6, 12), scaled_ray.origin))
    assert(np.allclose(vector(0, 3, 0), scaled_ray.direction))
