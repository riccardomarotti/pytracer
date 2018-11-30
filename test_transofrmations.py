import transformations
from tuples import point, vector
import numpy as np
from numpy.linalg import inv
import math


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


def test_rotating_a_point_around_the_x_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    full_quarter = transformations.rotation_x(math.pi/2)

    assert(point(0, math.sqrt(2)/2, math.sqrt(2)/2).all()
           == half_quarter.dot(p).all())
    assert(point(0, 0, 1).all() == full_quarter.dot(p).all())


def test_the_inverse_of_an_x_rotation_rotates_in_the_opposite_direction():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    inverse = inv(half_quarter)

    assert(point(0, math.sqrt(2)/2, -math.sqrt(2)/2).all()
           == inverse.dot(p).all())


def test_rotating_a_point_around_the_y_axis():
    p = point(0, 0, 1)
    half_quarter = transformations.rotation_y(math.pi/4)
    full_quarter = transformations.rotation_y(math.pi/2)

    assert(point(-math.sqrt(2)/2, 0, math.sqrt(2)/2).all()
           == half_quarter.dot(p).all())
    assert(point(1, 0, 0).all() == full_quarter.dot(p).all())


def test_rotating_a_point_around_the_z_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_z(math.pi/4)
    full_quarter = transformations.rotation_z(math.pi/2)

    assert(point(-math.sqrt(2)/2, math.sqrt(2)/2, 0).all()
           == half_quarter.dot(p).all())
    assert(point(-1, 0, 0).all() == full_quarter.dot(p).all())
