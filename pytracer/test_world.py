import numpy as np
from pytracer.world import World, default_world
from pytracer.rays import Ray
from pytracer.tuples import point, vector
from pytracer.intersections import Intersection
from pytracer.colors import color, black
from pytracer.lights import PointLight


def test_createing_a_world():
    w = World()

    assert(len(w) == 0)
    assert(w.light is None)


def test_intersect_a_world_with_a_ray():
    w = default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))

    xs = w.intersect(r)

    assert(len(xs) == 4)
    assert(xs[0].t == 4)
    assert(xs[1].t == 4.5)
    assert(xs[2].t == 5.5)
    assert(xs[3].t == 6)


def test_shading_an_intersection():
    w = default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = w[0]
    i = Intersection(4, shape)

    comps = i.prepare_computations(r)
    c = w.shade_hit(comps)

    assert(np.allclose(color(0.38066, 0.47583, 0.2855), c, atol=0.0001))


def test_shading_an_intersection_from_the_inside():
    w = default_world()
    w._light = PointLight(point(0, 0.25, 0), color(1, 1, 1))
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    shape = w[1]
    i = Intersection(0.5, shape)

    comps = i.prepare_computations(r)
    c = w.shade_hit(comps)

    assert(np.allclose(color(0.90498, 0.90498, 0.90498), c))


def test_the_color_when_the_ray_misses():
    w = default_world()
    r = Ray(point(0, 0, -5), vector(0, 1, 0))

    c = w.color_at(r)
    assert(np.allclose(black, c))


def test_the_color_when_the_ray_hits():
    w = default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))

    c = w.color_at(r)
    assert(np.allclose(color(0.38066, 0.47583, 0.2855), c, atol=0.0001))


def test_the_color_with_an_intersection_behind_the_ray():
    w = default_world()
    outer = w[0]
    outer._material._ambient = 1
    inner = w[1]
    inner._material._ambient = 1

    r = Ray(point(0, 0, 0.75), vector(0, 0, -1))

    c = w.color_at(r)
    assert(np.allclose(c, inner.material.color))


def test_there_is_no_shadow_when_nothing_is_collinear_with_point_and_light():
    w = default_world()
    p = point(0, 10, 0)

    assert(w.is_shadowed(p) == False)


def test_the_shadow_when_an_object_is_between_the_point_and_the_light():
    w = default_world()
    p = point(10, -10, 10)

    assert(w.is_shadowed(p))


def test_there_is_no_shadow_when_an_object_is_behind_the_light():
    w = default_world()
    p = point(-20, 20, -20)

    assert(w.is_shadowed(p) == False)


def test_there_is_no_shadow_when_an_object_is_behind_the_point():
    w = default_world()
    p = point(-2, 2, -2)

    assert(w.is_shadowed(p) == False)
