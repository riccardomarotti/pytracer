from tuples import point, vector
import rays
import numpy as np


def test_computing_a_point_from_a_distance():
    p = point(2, 3, 4)
    v = vector(1, 0, 0)

    assert(point(2, 3, 4).all() == rays.position(p, v, 0).all())
    assert(point(3, 3, 4).all() == rays.position(p, v, 1).all())
    assert(point(1, 3, 4).all() == rays.position(p, v, -1).all())
    assert(point(4.5, 3, 4).all() == rays.position(p, v, 2.5).all())
