from pytracer.transformations import identity_matrix
import math


class Camera:
    def __init__(self, hsize, vsize, field_of_view, transform=None):
        self._hsize = hsize
        self._vsize = vsize
        self._field_of_view = field_of_view
        if transform is None:
            self._transform = identity_matrix
        else:
            self._transform = transform

        half_view = math.tan(self._field_of_view / 2)
        aspect = self._hsize / self._vsize

        if aspect >= 1:
            self._half_width = half_view
            self._half_height = half_view / aspect
        else:
            self._half_width = half_view * aspect
            self._half_height = half_view

        self._pixel_size = (self._half_width * 2) / self._hsize

    @property
    def pixel_size(self):
        return self._pixel_size
