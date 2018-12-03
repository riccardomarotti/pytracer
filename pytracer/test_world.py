from pytracer.world import World, default_world
from pytracer.rays import Ray
from pytracer.tuples import point, vector


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
