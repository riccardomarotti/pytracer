import pytest
from pytracer.materials import Material
from pytracer.tuples import vector as vector, point
from pytracer.lights import PointLight
from pytracer.colors import color as color
import numpy as np
import math


@pytest.fixture()
def material():
    return Material()


@pytest.fixture()
def position():
    return point(0, 0, 0)


def test_defaultmaterial(material):
    assert(np.allclose([1, 1, 1], material.color))


def test_lighting_with_the_eye_between_the_light_and_the_surface(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 0, -10), color(1, 1, 1))

    result = material.lighting(light, position, eyev, normalv, False)

    assert(np.array_equal(color(1.9, 1.9, 1.9), result))


def test_lighting_with_the_eye_between_the_light_and_the_surface_and_wyw_offset_45_degrees(material, position):
    eyev = vector(0, math.sqrt(2)/2, -math.sqrt(2)/2)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 0, -10), color(1, 1, 1))

    result = material.lighting(light, position, eyev, normalv, False)

    assert(np.array_equal(color(1.0, 1.0, 1.0), result))


def test_lighting_with_the_eye_opposite_surface_light_offest_45_degrees(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 10, -10), color(1, 1, 1))

    result = material.lighting(light, position, eyev, normalv, False)

    assert(np.allclose(color(0.7364, 0.7364, 0.7364), result))


def test_lighting_with_the_eye_in_the_path_of_the_reflection_vector(material, position):
    eyev = vector(0, -math.sqrt(2)/2, -math.sqrt(2)/2)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 10, -10), color(1, 1, 1))

    result = material.lighting(light, position, eyev, normalv, False)

    assert(np.allclose(color(1.6364, 1.6364, 1.6364), result))


def test_lighting_with_the_light_behind_the_surface(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 0, 10), color(1, 1, 1))

    result = material.lighting(light, position, eyev, normalv, False)

    assert(np.array_equal(color(0.1, 0.1, 0.1), result))


def test_lighting_with_the_surface_in_shadow(material, position):
    eyev = vector(0, 0, -1)
    normalv = vector(0, 0, -1)
    light = PointLight(point(0, 0, -10), color(1, 1, 1))
    in_shadow = True

    result = material.lighting(light, position, eyev, normalv, in_shadow)
    assert(np.allclose(color(0.1, 0.1, 0.1), result))
