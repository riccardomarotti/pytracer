from pytracer import tuples
from pytracer.tuples import vector, point, reflect
import numpy as np
import math


def test_a_point_is_an_array_with_w_set_to_1():
    actual_point = tuples.point(4.3, -4.2, 3.1)
    expected_point = np.array([4.3, -4.2, 3.1, 1.0])

    assert(expected_point.all() == actual_point.all())


def test_a_vector_is_an_array_with_w_set_to_0():
    actual_point = tuples.vector(4.3, -4.2, 3.1)
    expected_point = np.array([4.3, -4.2, 3.1, 0])

    assert(expected_point.all() == actual_point.all())


def test_normalize():
    v = tuples.vector(4, 0, 0)
    assert(tuples.vector(1, 0, 0).all() == tuples.normalize(v).all())

    v = tuples.vector(1, 2, 3)
    assert(np.allclose(tuples.vector(0.26726, 0.53452, 0.80178), tuples.normalize(v)))


def test_reflecting_a_vector_approaching_at_45_degrees():
    v = vector(1, -1, 0)
    n = vector(0, 1, 0)

    r = reflect(v, n)

    assert(vector(1, 1, 0).all() == r.all())


def test_reflecting_a_vector_off_a_slanted_surface():
    v = vector(0, -1, 0)
    n = vector(math.sqrt(2)/2, math.sqrt(2)/2, 0)

    r = reflect(v, n)

    assert(vector(1, 0, 0).all() == r.all())
