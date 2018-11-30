import canvas
import numpy as np


def test_constructing_the_PPM_header():
    actualPPM = canvas.PPM(5, 3)

    expectedPPM = """P3
5 3
255"""
    assert(actualPPM.startswith(expectedPPM))


def test_constructing_the_PPM_pixel_data():
    c1 = [1.5, 0, 0]
    c2 = [0, 0.5, 0]
    c3 = [-0.5, 0, 1]

    pixels = np.zeros((5, 3, 3))
    pixels[0, 0] = c1
    pixels[2, 1] = c2
    pixels[4, 2] = c3

    actualPPM = canvas.PPM(5, 3, pixels)

    expectedPPM = """P3
5 3
255
255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 127 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"""

    assert(expectedPPM == actualPPM)


def test_splitting_long_lines_in_PPM():
    pixels = np.ones((10, 2, 3))*[1, 0.8, 0.6]
    actualPPM = canvas.PPM(10, 2, pixels)

    expectedPPM = """255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153
255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204
153 255 204 153 255 204 153 255 204 153 255 204 153"""

    assert(len(actualPPM.split("\n")) == 8)
    actualPPM = "\n".join(actualPPM.split("\n")[3:7])

    assert(expectedPPM == actualPPM)


def test_PPM_files_are_terminated_by_a_newline():
    ppm = canvas.PPM(5, 3)

    assert(ppm[-1] == '\n')
