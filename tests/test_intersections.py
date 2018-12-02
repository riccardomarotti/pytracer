from pytracer import intersections
import numpy as np


def test_hit_when_all_intersections_have_positive_t():
    xs = np.array([[1], [2]])

    i = intersections.hit(xs)

    assert(i.all() == xs[0].all())


def test_hit_when_some_intersections_have_negative_t():
    xs = np.array([[-1], [1]])

    i = intersections.hit(xs)

    assert(i.all() == xs[1].all())


def test_hit_when_all_intersections_are_negative():
    xs = np.array([[-2], [-1]])

    i = intersections.hit(xs)

    assert(len(xs == 0))


def test_hit_is_always_the_lowest_non_negative_intersection():
    xs = np.array([[5], [7], [-3], [2]])

    i = intersections.hit(xs)

    assert(i.all() == xs[3].all())
