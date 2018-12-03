import pytest
import pytracer.materials as materials
from pytracer.tuples import vector as vector, point
from pytracer.lights import point_light as point_light
from pytracer.colors import color as color
import numpy as np
import math


@pytest.fixture()
def material():
    return materials.material()


@pytest.fixture()
def position():
    return point(0, 0, 0)


def test_defaultmaterial():
    m = materials.material()

    assert(np.allclose([1, 1, 1], materials.color(m)))


def test_lighting_with_the_eye_between_the_light_and_the_surface(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = point_light(point(0, 0, -10), color(1, 1, 1))

    result = materials.lighting(material, light, position, eyev, normalv)

    assert(np.array_equal(color(1.9, 1.9, 1.9), result))


def test_lighting_with_the_eye_between_the_light_and_the_surface_and_wyw_offset_45_degrees(material, position):
    eyev = vector(0, math.sqrt(2)/2, -math.sqrt(2)/2)
    normalv = vector(0, 0, -1)
    light = point_light(point(0, 0, -10), color(1, 1, 1))

    result = materials.lighting(material, light, position, eyev, normalv)

    assert(np.array_equal(color(1.0, 1.0, 1.0), result))


def test_lighting_with_the_eye_opposite_surface_light_offest_45_degrees(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = point_light(point(0, 10, -10), color(1, 1, 1))

    result = materials.lighting(material, light, position, eyev, normalv)

    assert(np.allclose(color(0.7364, 0.7364, 0.7364), result))


def test_lighting_with_the_eye_in_the_path_of_the_reflection_vector(material, position):
    eyev = vector(0, -math.sqrt(2)/2, -math.sqrt(2)/2)
    normalv = vector(0, 0, -1)
    light = point_light(point(0, 10, -10), color(1, 1, 1))

    result = materials.lighting(material, light, position, eyev, normalv)

    assert(np.allclose(color(1.6364, 1.6364, 1.6364), result))


def test_lighting_with_the_light_behind_the_surface(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = point_light(point(0, 0, 10), color(1, 1, 1))

    result = materials.lighting(material, light, position, eyev, normalv)

    assert(np.array_equal(color(0.1, 0.1, 0.1), result))
