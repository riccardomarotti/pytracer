from pytracer import spheres
from pytracer import tuples
from pytracer import transformations
from pytracer import rays
from pytracer import materials as materials
import math
from pytracer.tuples import point, vector
import numpy as np


def test_identity_is_a_sphere_default_transformation():
    sphere = spheres.sphere()
    expected_transformation = transformations.identity_matrix
    actual_transformation = spheres.transformation(sphere)

    assert(np.array_equal(expected_transformation(), actual_transformation()))


def test_a_ray_intersects_a_sphere_at_two_points():
    origin = point(0, 0, -5)
    direction = vector(0, 0, 1)

    xs = spheres.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == 4.0)
    assert(xs[1] == 6.0)


def test_a_ray_intersects_a_sphere_at_a_tangent():
    origin = point(0, 1, -5)
    direction = vector(0, 0, 1)

    xs = spheres.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == 5.0)
    assert(xs[1] == 5.0)


def test_a_ray_misses_a_sphere():
    origin = point(0, 2, -5)
    direction = vector(0, 0, 1)

    xs = spheres.intersect(origin, direction)

    assert(len(xs) == 0)


def test_a_ray_originates_inside_a_sphere():
    origin = point(0, 0, 0)
    direction = vector(0, 0, 1)

    xs = spheres.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == -1.0)
    assert(xs[1] == 1.0)


def test_a_sphere_behind_a_ary():
    origin = point(0, 0, 5)
    direction = vector(0, 0, 1)

    xs = spheres.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == -6.0)
    assert(xs[1] == -4.0)


def test_intersecting_a_scaled_sphere_with_a_ray():
    sphere_transform = transformations.scaling(2, 2, 2)
    ray_origin = point(0, 0, -5)
    ray_direction = vector(0, 0, 1)

    xs = spheres.intersect(ray_origin, ray_direction, sphere_transform)

    assert(xs[0] == 3)
    assert(xs[1] == 7)


def test_intersecting_a_translated_sphere_with_a_ray():
    sphere_transform = transformations.translation(5, 0, 0)
    ray_origin = point(0, 0, -5)
    ray_direction = vector(0, 0, 1)

    xs = spheres.intersect(ray_origin, ray_direction, sphere_transform)

    assert(len(xs) == 0)


def test_normal_on_a_sphere_at_a_point_on_the_x_axis():
    n = spheres.normal_at(point(1, 0, 0))

    assert((vector(1, 0, 0) == n).all())


def test_normal_on_a_sphere_at_a_point_on_the_y_axis():
    n = spheres.normal_at(point(0, 1, 0))

    assert((vector(0, 1, 0) == n).all())


def test_normal_on_a_sphere_at_a_point_on_the_z_axis():
    n = spheres.normal_at(point(0, 0, 1))

    assert((vector(0, 0, 1) == n).all())


def test_normal_on_a_sphere_at_a_non_axial_point():
    p = math.sqrt(3)/3
    n = spheres.normal_at(point(p, p, p))

    assert(np.allclose(vector(p, p, p), n))


def test_the_normal_is_a_normalized_vector():
    p = math.sqrt(3)/3
    n = spheres.normal_at(point(p, p, p))

    assert(np.allclose(tuples.normalize(n), n))


def test_computing_the_normal_on_a_translated_sphere():
    transform = transformations.translation(0, 1, 0)

    n = spheres.normal_at(point(0, 1.70711, -0.70711), transform)
    assert(np.allclose(vector(0, 0.70711, -0.70711), n))


def test_computing_the_normal_on_a_transformed_sphere():
    s = transformations.scaling(1, 0.5, 1)
    r = transformations.rotation_z(math.pi/5)
    transform = transformations.concat(s, r)

    n = spheres.normal_at(point(0, math.sqrt(2)/2, -math.sqrt(2)/2), transform)
    assert(np.allclose(vector(0, 0.97014, -0.242535), n))


def test_a_sphere_has_a_material():
    s = spheres.sphere()
    expected_material = materials.material()
    actual_material = spheres.material(s)

    assert(np.array_equal(expected_material, actual_material))


def test_a_sphere_may_be_asigned_a_material():
    m = materials.material(ambient=1)

    s = spheres.sphere(material=m)
    actual_material = spheres.material(s)

    assert(np.array_equal(m, actual_material))
