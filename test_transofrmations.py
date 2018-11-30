import transformations
from tuples import point, vector
import numpy as np
from numpy.linalg import inv


def test_multiplying_by_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    p = point(-3, 4, 5)

    assert(point(2, 1, 7).all() == transform.dot(p).all())


def test_multiplying_by_the_inverse_of_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    transform = inv(transform)
    p = point(-3, 4, 5)

    assert(point(-8, 7, 3).all() == transform.dot(p).all())


def test_translation_does_not_affect_vectors():
    transform = transformations.translation(5, -3, 2)
    v = vector(-3, 4, 5)

    assert(v.all() == transform.dot(v).all())


def test_a_scaling_matrix_applied_to_a_point():
    transform = transformations.scaling(2, 3, 4)
    p = point(-4, 6, 8)

    assert(point(-8, 18, 32).all() == (transform.dot(p)).all())


def test_a_scaling_matrix_applied_to_a_vecor():
    transform = transformations.scaling(2, 3, 4)
    v = vector(-4, 6, 8)

    assert(vector(-8, 18, 32).all() == (transform.dot(v)).all())


def test_multiplying_the_inverse_of_a_scaling_matrix():
    transform = transformations.scaling(2, 3, 4)
    inverse = inv(transform)
    v = vector(-4, 6, 8)

    expected = vector(-2, 2, 2)
    actual = inverse.dot(v)
    assert(expected.all() == actual.all())


def test_reflection_is_scaling_by_a_negative_value():
    transform = transformations.scaling(-1, 1, 1)
    p = point(2, 3, 4)

    assert(point(-2, 3, 4).all() == (transform.dot(p)).all())
