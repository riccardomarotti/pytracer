import numpy as np
import math
import pytracer.lights as lights
from pytracer.tuples import normalize as normalize
from pytracer.tuples import reflect as reflect
from pytracer.colors import black, white


class Material:
    def __init__(self, color=None, ambient=0.1, diffuse=0.9, specular=0.9, shininess=200):
        if color is None:
            self._color = white
        else:
            self._color = color

        self._ambient = ambient
        self._diffuse = diffuse
        self._specular = specular
        self._shininess = shininess

    @property
    def color(self):
        return self._color

    @property
    def ambient(self):
        return self._ambient

    @property
    def diffuse(self):
        return self._diffuse

    @property
    def specular(self):
        return self._specular

    @property
    def shininess(self):
        return self._shininess

    def lighting(self, light, point, eye_vector, normal_vector):
        effective_color = self.color * light.intensity
        light_vector = normalize(light.position - point)
        amb = effective_color * self.ambient
        light_dot_normal = light_vector.dot(normal_vector)
        diff = black
        spec = black
        if light_dot_normal >= 0:
            diff = effective_color * self.diffuse * light_dot_normal

            reflect_vector = reflect(-light_vector, normal_vector)
            reflect_dot_eye = reflect_vector.dot(eye_vector)

            if reflect_dot_eye > 0:
                factor = math.pow(reflect_dot_eye, self.shininess)
                spec = light.intensity * self.specular * factor

        return amb + diff + spec
