import pytracer.lights as lights
from pytracer.tuples import point
from pytracer.colors import color as color
import numpy as np


def test_a_point_light_has_a_position_and_intensity():
    position = point(0, 0, 0)
    intensity = color(1, 1, 1)
    light = lights.point_light(position, intensity)

    assert(np.allclose(position, lights.position(light)))
    assert(np.allclose(intensity, lights.intensity(light)))
