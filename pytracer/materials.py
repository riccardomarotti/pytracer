import numpy as np


def default():
    return np.array([[1, 1, 1], 0.1, 0.9, 0.9, 200])


def color(material):
    return material[0]


def ambient(material):
    return material[1]


def diffuse(material):
    return material[2]


def specular(material):
    return material[3]


def shininess(material):
    return material[4]
