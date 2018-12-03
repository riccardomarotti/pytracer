from pytracer.spheres import Sphere
from pytracer import tuples
from pytracer import transformations
from pytracer.rays import Ray
from pytracer.materials import Material
import math
from pytracer.tuples import point, vector
import numpy as np


def test_identity_is_a_sphere_default_transformation():
    sphere = Sphere()
    expected_transformation = transformations.identity_matrix
    actual_transformation = sphere.transformation

    assert(np.array_equal(expected_transformation(), actual_transformation()))


def test_a_ray_intersects_a_sphere_at_two_points():
    xs = Sphere().intersect(Ray(point(0, 0, -5), vector(0, 0, 1)))

    assert(len(xs) == 2)
    assert(xs[0].t == 4.0)
    assert(xs[1].t == 6.0)


def test_a_ray_intersects_a_sphere_at_a_tangent():
    xs = Sphere().intersect(Ray(point(0, 1, -5), vector(0, 0, 1)))

    assert(len(xs) == 2)
    assert(xs[0].t == 5.0)
    assert(xs[1].t == 5.0)


def test_a_ray_misses_a_sphere():
    xs = Sphere().intersect(Ray(point(0, 2, -5), vector(0, 0, 1)))

    assert(len(xs) == 0)


def test_a_ray_originates_inside_a_sphere():
    xs = Sphere().intersect(Ray(point(0, 0, 0), vector(0, 0, 1)))

    assert(len(xs) == 2)
    assert(xs[0].t == -1.0)
    assert(xs[1].t == 1.0)


def test_a_sphere_behind_a_ary():
    xs = Sphere().intersect(Ray(point(0, 0, 5), vector(0, 0, 1)))

    assert(len(xs) == 2)
    assert(xs[0].t == -6.0)
    assert(xs[1].t == -4.0)


def test_intersecting_a_scaled_sphere_with_a_ray():
    sphere = Sphere(transformation=transformations.scaling(2, 2, 2))
    ray = Ray(point(0, 0, -5), vector(0, 0, 1))

    xs = sphere.intersect(ray)

    assert(xs[0].t == 3)
    assert(xs[1].t == 7)


def test_intersecting_a_translated_sphere_with_a_ray():
    sphere = Sphere(transformation=transformations.translation(5, 0, 0))
    ray = Ray(point(0, 0, -5), vector(0, 0, 1))

    xs = sphere.intersect(ray)

    assert(len(xs) == 0)


def test_normal_on_a_sphere_at_a_point_on_the_x_axis():
    n = Sphere().normal_at(point(1, 0, 0))

    assert((vector(1, 0, 0) == n).all())


def test_normal_on_a_sphere_at_a_point_on_the_y_axis():
    n = Sphere().normal_at(point(0, 1, 0))

    assert((vector(0, 1, 0) == n).all())


def test_normal_on_a_sphere_at_a_point_on_the_z_axis():
    n = Sphere().normal_at(point(0, 0, 1))

    assert((vector(0, 0, 1) == n).all())


def test_normal_on_a_sphere_at_a_non_axial_point():
    p = math.sqrt(3)/3
    n = Sphere().normal_at(point(p, p, p))

    assert(np.allclose(vector(p, p, p), n))


def test_the_normal_is_a_normalized_vector():
    p = math.sqrt(3)/3
    n = Sphere().normal_at(point(p, p, p))

    assert(np.allclose(tuples.normalize(n), n))


def test_computing_the_normal_on_a_translated_sphere():
    sphere = Sphere(transformations.translation(0, 1, 0))

    n = sphere.normal_at(point(0, 1.70711, -0.70711))
    assert(np.allclose(vector(0, 0.70711, -0.70711), n))


def test_computing_the_normal_on_a_transformed_sphere():
    s = transformations.scaling(1, 0.5, 1)
    r = transformations.rotation_z(math.pi/5)
    transform = transformations.concat(s, r)

    sphere = Sphere(transform)

    n = sphere.normal_at(point(0, math.sqrt(2)/2, -math.sqrt(2)/2))
    assert(np.allclose(vector(0, 0.97014, -0.242535), n))


def test_a_sphere_has_a_material():
    expected_material = Material()
    s = Sphere(material=expected_material)
    actual_material = s.material

    assert(expected_material is actual_material)


def test_a_sphere_may_be_asigned_a_material():
    m = Material(ambient=1)

    s = Sphere(material=m)
    actual_material = s.material

    assert(np.array_equal(m, actual_material))
