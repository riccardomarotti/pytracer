from pytracer.lights import PointLight
from pytracer.tuples import point
from pytracer.colors import color as color
import numpy as np


def test_a_point_light_has_a_position_and_intensity():
    position = point(0, 0, 0)
    intensity = color(1, 1, 1)
    light = PointLight(position, intensity)

    assert(np.allclose(position, light.position))
    assert(np.allclose(intensity, light.intensity))
