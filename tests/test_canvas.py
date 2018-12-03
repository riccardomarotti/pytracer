from pytracer.canvas import Canvas
from pytracer.colors import color
import numpy as np


def test_constructing_the_PPM_header():
    actualPPM = Canvas(5, 3).to_PPM()

    expectedPPM = """P3
5 3
255"""
    assert(actualPPM.startswith(expectedPPM))


def test_constructing_the_PPM_pixel_data():
    canvas = Canvas(5, 3)

    c1 = color(1.5, 0, 0)
    c2 = color(0, 0.5, 0)
    c3 = color(-0.5, 0, 1)

    canvas.write_pixel(0, 0, c1)
    canvas.write_pixel(2, 1, c2)
    canvas.write_pixel(4, 2, c3)

    actualPPM = canvas.to_PPM()

    expectedPPM = """P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 127 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"""

    assert(expectedPPM == actualPPM)


def test_splitting_long_lines_in_PPM():
    canvas = Canvas(10, 2, np.ones((10, 2, 3))*[1, 0.8, 0.6])
    actualPPM = canvas.to_PPM()

    expectedPPM = """255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153"""

    assert(len(actualPPM.split("\n")) == 8)
    actualPPM = "\n".join(actualPPM.split("\n")[3:7])

    assert(expectedPPM == actualPPM)


def test_PPM_files_are_terminated_by_a_newline():
    ppm = Canvas(5, 3).to_PPM()

    assert(ppm[-1] == '\n')
