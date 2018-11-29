from canvas import Canvas
import numpy as np


def test_creating_canvas():
    c = Canvas(10, 20)
    expected_pixels = np.zeros((20, 10, 3))

    assert(c.width == 10)
    assert(c.height == 20)
    assert(c.pixels().all() == expected_pixels.all())
