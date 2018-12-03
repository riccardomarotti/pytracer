import math
import numpy as np
from pytracer.camera import Camera
from pytracer.transformations import identity_matrix


def test_constructing_a_camera():
    hsize = 160
    vsize = 120
    field_of_view = math.pi/2

    c = Camera(hsize, vsize, field_of_view)

    assert(c._hsize == 160)
    assert(c._vsize == 120)
    assert(c._field_of_view == math.pi/2)
    assert(np.array_equal(c._transform(identity_matrix()), identity_matrix()))
