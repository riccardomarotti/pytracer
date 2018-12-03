from pytracer import transformations
from pytracer.transformations import identity_matrix, view_transformation, scaling, translation
from pytracer.tuples import point, vector
import numpy as np
from numpy.linalg import inv
import math


def test_multiplying_by_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    p = point(-3, 4, 5)

    assert((point(2, 1, 7) == transform(p)).all())


def test_multiplying_by_the_inverse_of_a_translation_matrix():
    transform = transformations.translation(5, -3, 2)
    transform = transformations.invert(transform)
    p = point(-3, 4, 5)

    assert((point(-8, 7, 3) == transform(p)).all())


def test_translation_does_not_affect_vectors():
    transform = transformations.translation(5, -3, 2)
    v = vector(-3, 4, 5)

    assert((v == transform(v)).all())


def test_a_scaling_matrix_applied_to_a_point():
    transform = transformations.scaling(2, 3, 4)
    p = point(-4, 6, 8)

    assert((point(-8, 18, 32) == (transform(p))).all())


def test_a_scaling_matrix_applied_to_a_vecor():
    transform = transformations.scaling(2, 3, 4)
    v = vector(-4, 6, 8)

    assert((vector(-8, 18, 32) == (transform(v))).all())


def test_multiplying_the_inverse_of_a_scaling_matrix():
    transform = transformations.scaling(2, 3, 4)
    inverse = transformations.invert(transform)
    v = vector(-4, 6, 8)

    expected = vector(-2, 2, 2)
    actual = inverse(v)
    assert((expected == actual).all())


def test_reflection_is_scaling_by_a_negative_value():
    transform = transformations.scaling(-1, 1, 1)
    p = point(2, 3, 4)

    assert((point(-2, 3, 4) == (transform(p))).all())


def test_rotating_a_point_around_the_x_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    full_quarter = transformations.rotation_x(math.pi/2)

    assert(np.allclose(point(0, math.sqrt(2)/2, math.sqrt(2)/2), half_quarter(p)))
    assert(np.allclose(point(0, 0, 1), full_quarter(p)))


def test_the_inverse_of_an_x_rotation_rotates_in_the_opposite_direction():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_x(math.pi/4)
    inverse = transformations.invert(half_quarter)

    assert(np.allclose(point(0, math.sqrt(2)/2, -math.sqrt(2)/2), inverse(p)))


def test_rotating_a_point_around_the_y_axis():
    p = point(0, 0, 1)
    half_quarter = transformations.rotation_y(math.pi/4)
    full_quarter = transformations.rotation_y(math.pi/2)

    assert(np.allclose(point(math.sqrt(2)/2, 0, math.sqrt(2)/2), half_quarter(p)))
    assert(np.allclose(point(1, 0, 0), full_quarter(p)))


def test_rotating_a_point_around_the_z_axis():
    p = point(0, 1, 0)
    half_quarter = transformations.rotation_z(math.pi/4)
    full_quarter = transformations.rotation_z(math.pi/2)

    assert(np.allclose(point(-math.sqrt(2)/2, math.sqrt(2)/2, 0), half_quarter(p)))
    assert(np.allclose(point(-1, 0, 0), full_quarter(p)))


def test_a_shearing_transformation_moves_x_in_proportion_to_y():
    transform = transformations.shearing(1, 0, 0, 0, 0, 0)
    assert((point(5, 3, 4) == transform(point(2, 3, 4))).all())


def test_a_shearing_transformation_moves_x_in_proportion_to_z():
    transform = transformations.shearing(0, 1, 0, 0, 0, 0)
    assert((point(6, 3, 4) == transform(point(2, 3, 4))).all())


def test_a_shearing_transformation_moves_y_in_proportion_to_x():
    transform = transformations.shearing(0, 0, 1, 0, 0, 0)
    assert((point(2, 5, 4) == transform(point(2, 3, 4))).all())


def test_a_shearing_transformation_moves_y_in_proportion_to_z():
    transform = transformations.shearing(0, 0, 0, 1, 0, 0)
    assert((point(2, 7, 4) == transform(point(2, 3, 4))).all())


def test_a_shearing_transformation_moves_z_in_proportion_to_x():
    transform = transformations.shearing(0, 0, 0, 0, 1, 0)
    assert((point(2, 3, 6) == transform(point(2, 3, 4))).all())


def test_a_shearing_transformation_moves_z_in_proportion_to_y():
    transform = transformations.shearing(0, 0, 0, 0, 0, 1)
    assert((point(2, 3, 7) == transform(point(2, 3, 4))).all())


def test_individual_transformations_are_applied_in_sequence():
    p = point(1, 0, 1)
    A = transformations.rotation_x(math.pi/2)
    B = transformations.scaling(5, 5, 5)
    C = transformations.translation(10, 5, 7)

    p2 = A(p)
    assert(np.allclose(point(1, -1, 0), p2))

    p3 = B(p2)
    assert(np.allclose(point(5, -5, 0), p3))

    p4 = C(p3)
    assert(np.allclose(point(15, 0, 7), p4))


def test_chained_transofrmations_must_be_applied_in_reverse_order():
    p = point(1, 0, 1)
    A = transformations.rotation_x(math.pi/2)
    B = transformations.scaling(5, 5, 5)
    C = transformations.translation(10, 5, 7)

    CBA = transformations.concat(C, B, A)

    assert(np.allclose(point(15, 0, 7), CBA(p)))


def test_the_view_transformation_matrix_for_the_default_orientation():
    from_ = point(0, 0, 0)
    to = point(0, 0, -1)
    up = vector(0, 1, 0)

    t = view_transformation(from_, to, up)

    actual_transform_matrix = t(identity_matrix())
    expected_transform_matrix = identity_matrix()

    assert(np.allclose(actual_transform_matrix, expected_transform_matrix))


def test_a_transformation_matrix_looking_in_positive_z_direction():
    from_ = point(0, 0, 0)
    to = point(0, 0, 1)
    up = vector(0, 1, 0)

    t = view_transformation(from_, to, up)

    actual_transform_matrix = t(identity_matrix())
    expected_transform_matrix = scaling(-1, 1, -1)(identity_matrix())

    assert(np.allclose(actual_transform_matrix, expected_transform_matrix))


def test_the_view_transformation_moves_the_world():
    from_ = point(0, 0, 8)
    to = point(0, 0, 0)
    up = vector(0, 1, 0)

    t = view_transformation(from_, to, up)

    actual_transform_matrix = t(identity_matrix())
    expected_transform_matrix = translation(0, 0, -8)(identity_matrix())

    assert(np.allclose(actual_transform_matrix, expected_transform_matrix))


def test_an_arbitrary_view_transformation():
    from_ = point(1, 3, 2)
    to = point(4, -2, 8)
    up = vector(1, 1, 0)

    view_transform = view_transformation(from_, to, up)

    actual_transform_matrix = view_transform(identity_matrix())
    expected_transform_matrix = np.array([
        [-0.50709, 0.50709, 0.67612, -2.36643],
        [0.76772, 0.60609, 0.12122, -2.82843],
        [-0.35857, 0.59761, -0.71714, 0],
        [0, 0, 0, 1],
    ])

    # assert(actual_transform_matrix[0][0] == expected_transform_matrix[0][0])
    assert(np.allclose(actual_transform_matrix,
                       expected_transform_matrix, atol=0.0001))
