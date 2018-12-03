from pytracer.world import World


def test_createing_a_world():
    w = World()

    assert(len(w) == 0)
    assert(w.light is None)
