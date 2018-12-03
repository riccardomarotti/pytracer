from pytracer.spheres import Sphere
from pytracer.rays import Ray
from pytracer.tuples import point, vector
from pytracer.intersections import Intersection, Intersections
from pytracer.intersections import hit_fast
import numpy as np


def test_an_intersection_encapsulates_t_and_object():
    s = Sphere()
    i = Intersection(3.5, s)

    assert(i.object is s)
    assert(i.t == 3.5)


def test_aggregating_intersections():
    s = Sphere()
    i1 = Intersection(1, s)
    i2 = Intersection(2, s)
    xs = Intersections(i1, i2)

    assert(xs[0].t == 1)
    assert(xs[1].t == 2)


def test_intersect_sets_the_object_on_the_intersection():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    s = Sphere()

    xs = s.intersect(r)

    assert(len(xs) == 2)
    assert(xs[0].object is s)
    assert(xs[1].object is s)


def test_hit_is_empty_when_input_is_empty():
    xs = Intersections()

    i = xs.hit()

    assert(i is None)


def test_hit_when_all_intersections_have_positive_t():
    s = Sphere()
    i1 = Intersection(1.0, s)
    i2 = Intersection(2.0, s)
    xs = Intersections(i2, i1)

    i = xs.hit()

    assert(i is i1)


def test_hit_when_some_intersections_have_negative_t():
    s = Sphere()
    i1 = Intersection(-1, s)
    i2 = Intersection(1, s)
    xs = Intersections(i2, i1)

    i = xs.hit()

    assert(i is i2)


def test_hit_when_all_intersections_are_negative():
    s = Sphere()
    i1 = Intersection(-2, s)
    i2 = Intersection(-1, s)
    xs = Intersections(i2, i1)

    i = xs.hit()

    assert(i is None)


def test_hit_is_always_the_lowest_non_negative_intersection():
    s = Sphere()
    i1 = Intersection(5.0, s)
    i2 = Intersection(7.0, s)
    i3 = Intersection(-3, s)
    i4 = Intersection(2, s)
    xs = Intersections(i1, i2, i3, i4)

    i = xs.hit()

    assert(i is i4)


def test_precompute_the_state_of_an_intersection():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = Sphere()
    i = Intersection(4, shape)

    comps = i.prepare_computations(r)

    assert(comps.t == i.t)
    assert(comps.object is i.object)
    assert(np.array_equal(point(0, 0, -1), comps.point))
    assert(np.array_equal(vector(0, 0, -1), comps.eyev))
    assert(np.array_equal(vector(0, 0, -1), comps.normalv))


def test_the_hit_when_an_intersection_occurs_on_the_outside():
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    shape = Sphere()
    i = Intersection(4, shape)

    comps = i.prepare_computations(r)

    assert(comps.inside == False)


def test_the_hit_when_an_intersection_occurs_on_the_inside():
    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    shape = Sphere()
    i = Intersection(1, shape)

    comps = i.prepare_computations(r)

    assert(np.array_equal(point(0, 0, 1), comps.point))
    assert(np.array_equal(vector(0, 0, -1), comps.eyev))
    assert(comps.inside == True)
    assert(np.array_equal(vector(0, 0, -1), comps.normalv))
