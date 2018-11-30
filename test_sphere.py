import sphere
from tuples import point, vector


def test_a_ray_intersects_a_sphere_at_two_points():
    origin = point(0, 0, -5)
    direction = vector(0, 0, 1)

    xs = sphere.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == 4.0)
    assert(xs[1] == 6.0)


def test_a_ray_intersects_a_sphere_at_a_tangent():
    origin = point(0, 1, -5)
    direction = vector(0, 0, 1)

    xs = sphere.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == 5.0)
    assert(xs[1] == 5.0)


def test_a_ray_misses_a_sphere():
    origin = point(0, 2, -5)
    direction = vector(0, 0, 1)

    xs = sphere.intersect(origin, direction)

    assert(len(xs) == 0)


def test_a_ray_originates_inside_a_sphere():
    origin = point(0, 0, 0)
    direction = vector(0, 0, 1)

    xs = sphere.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == -1.0)
    assert(xs[1] == 1.0)


def test_a_sphere_behind_a_ary():
    origin = point(0, 0, 5)
    direction = vector(0, 0, 1)

    xs = sphere.intersect(origin, direction)

    assert(len(xs) == 2)
    assert(xs[0] == -6.0)
    assert(xs[1] == -4.0)