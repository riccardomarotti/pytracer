import numpy as np
import math
import pytracer.lights as lights
from pytracer.tuples import normalize as normalize
from pytracer.tuples import reflect as reflect
import pytracer.colors as colors


def material(color=[1, 1, 1], ambient=0.1, diffuse=0.9, specular=0.9, shininess=200):
    return np.array([color, ambient, diffuse, specular, shininess])


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


def lighting(material, light, point, eye_vector, normal_vector):
    effective_color = color(material) * lights.intensity(light)
    light_vector = normalize(lights.position(light) - point)
    amb = effective_color * ambient(material)
    light_dot_normal = light_vector.dot(normal_vector)
    if light_dot_normal < 0:
        diff = colors.black()
        spec = colors.black()
    else:
        diff = effective_color * \
            diffuse(material) * light_dot_normal

        reflect_vector = reflect(-light_vector, normal_vector)
        reflect_dot_eye = reflect_vector.dot(eye_vector)

        if reflect_dot_eye <= 0:
            spec = colors.black()
        else:
            factor = math.pow(reflect_dot_eye, shininess(material))
            spec = lights.intensity(
                light) * specular(material) * factor

    return amb + diff + spec
