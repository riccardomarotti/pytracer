import tuples
import numpy as np


def test_a_point_is_an_array_with_w_set_to_1():
    actual_point = tuples.point(4.3, -4.2, 3.1)
    expected_point = np.array([4.3, -4.2, 3.1, 1.0])

    assert(expected_point.all() == actual_point.all())


def test_a_vector_is_an_array_with_w_set_to_0():
    actual_point = tuples.vector(4.3, -4.2, 3.1)
    expected_point = np.array([4.3, -4.2, 3.1, 0])

    assert(expected_point.all() == actual_point.all())
