from pytracer.lights import PointLight
from pytracer.tuples import point, vector
from pytracer.colors import color, white
from pytracer.spheres import Sphere
from pytracer.materials import Material
from pytracer.transformations import scaling
from pytracer.intersections import Intersection, Intersections


class World:
    def __init__(self, light=None, *objects):
        self._light = light
        self._objects = objects

    @property
    def light(self):
        return self._light

    def __len__(self):
        return len(self._objects)

    def intersect(self, ray):
        xs = []
        for obj in self._objects:
            xs.extend(obj.intersect(ray))

        xs.sort(key=lambda i: i.t)

        return Intersections(*xs)


def default_world():
    light = PointLight(point(-10, 10, -10), white())
    m = Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)
    s1 = Sphere(material=m)
    s2 = Sphere(transformation=scaling(0.5, 0.5, 0.5))

    return World(list, s1, s2)
