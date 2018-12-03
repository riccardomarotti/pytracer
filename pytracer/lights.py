import numpy as np


def point_light(position, intensity):
    return np.array([position, intensity])


def position(light):
    return light[0]


def intensity(light):
    return light[1]
