from pytracer.tuples import point, vector
import pytracer.transformations as transformations
import pytracer.rays as rays
import numpy as np


def test_computing_a_point_from_a_distance():
    origin = point(2, 3, 4)
    direction = vector(1, 0, 0)

    assert((point(2, 3, 4) == rays.position(origin, direction, 0)).all())
    assert((point(3, 3, 4) == rays.position(origin, direction, 1)).all())
    assert((point(1, 3, 4) == rays.position(origin, direction, -1)).all())
    assert((point(4.5, 3, 4) == rays.position(
        origin, direction, 2.5)).all())


def test_translating_a_ray():
    origin = point(1, 2, 3)
    direction = vector(0, 1, 0)

    m = transformations.translation(3, 4, 5)
    translated_origin, translated_direction = rays.apply(m, origin, direction)

    assert(np.allclose(point(4, 6, 8), translated_origin))
    assert(np.allclose(vector(0, 1, 0), translated_direction))


def test_scaling_a_ray():
    origin = point(1, 2, 3)
    direction = vector(0, 1, 0)

    m = transformations.scaling(2, 3, 4)
    scaled_origin, scaled_direction = rays.apply(m, origin, direction)

    assert(np.allclose(point(2, 6, 12), scaled_origin))
    assert(np.allclose(vector(0, 3, 0), scaled_direction))
