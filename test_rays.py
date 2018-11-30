from tuples import point, vector
import rays
import numpy as np


def test_computing_a_point_from_a_distance():
    origin = point(2, 3, 4)
    direction = vector(1, 0, 0)

    assert(point(2, 3, 4).all() == rays.position(origin, direction, 0).all())
    assert(point(3, 3, 4).all() == rays.position(origin, direction, 1).all())
    assert(point(1, 3, 4).all() == rays.position(origin, direction, -1).all())
    assert(point(4.5, 3, 4).all() == rays.position(
        origin, direction, 2.5).all())
