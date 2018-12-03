from pytracer.lights import PointLight
from pytracer.tuples import point, vector
from pytracer.colors import color, white, black
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

    def __getitem__(self, key):
        return self._objects[key]

    def __len__(self):
        return len(self._objects)

    def intersect(self, ray):
        xs = []
        for obj in self._objects:
            xs.extend(obj.intersect(ray))

        xs.sort(key=lambda i: i.t)

        return Intersections(*xs)

    def shade_hit(self, comps):
        return comps.object.material.lighting(self.light, comps.point, comps.eyev, comps.normalv)

    def color_at(self, ray):
        color = black

        xs = self.intersect(ray)
        intersection = xs.hit()
        if intersection is not None:
            comps = intersection.prepare_computations(ray)
            color = self.shade_hit(comps)

        return color


def default_world():
    light = PointLight(point(-10, 10, -10), white)
    m = Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2)
    s1 = Sphere(material=m)
    s2 = Sphere(transformation=scaling(0.5, 0.5, 0.5))

    return World(light, s1, s2)
