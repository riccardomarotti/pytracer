import numpy as np


def color(r, g, b):
    return np.array([r, g, b])


def black():
    return color(0., 0., 0.)


def white():
    return color(1., 1., 1.)
