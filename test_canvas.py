from canvas import Canvas
import numpy as np


def test_creating_canvas():
    c = Canvas(10, 20)
    expected_pixels = np.zeros((20, 10, 3))

    assert(c.width == 10)
    assert(c.height == 20)
    assert(c.pixels().all() == expected_pixels.all())


def test_constructing_the_PPM_header():
    c = Canvas(5, 3)

    expectedPPM = """P3
5 3
255"""
    assert(c.PPM().startswith(expectedPPM))


def test_constructing_the_PPM_pixel_data():
    c1 = [1.5, 0, 0]
    c2 = [0, 0.5, 0]
    c3 = [-0.5, 0, 1]

    pixels = np.zeros((5, 3, 3))
    pixels[0, 0] = c1
    pixels[2, 1] = c2
    pixels[4, 2] = c3

    c = Canvas(5, 3, pixels)

    expectedPPM = """P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 127 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"""

    assert(expectedPPM == c.PPM())
