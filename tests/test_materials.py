import pytracer.materials as materials
import numpy as np


def test_defaultmaterial():
    m = materials.default()

    assert(np.allclose([1, 1, 1], materials.color(m)))
