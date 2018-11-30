import transformations
from tuples import point, vector
import numpy as np


def test_multiplying_by_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    p = point(-3, 4, 5)

    assert(point(2, 1, 7).all() == transform.dot(p).all())


def test_multiplying_by_the_inverse_of_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    transform = np.invert(transform)
    p = point(-3, 4, 5)

    assert(point(-8, 7, 3).all() == transform.dot(p).all())


def test_translation_does_not_affect_vectors():
    transform = transformations.translation(5, -3, 2)
    v = vector(-3, 4, 5)

    assert(v.all() == transform.dot(v).all())
