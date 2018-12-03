from pytracer.transformations import identity_matrix


class Camera:
    def __init__(self, hsize, vsize, field_of_view, transform=None):
        self._hsize = hsize
        self._vsize = vsize
        self._field_of_view = field_of_view
        if transform is None:
            self._transform = identity_matrix
        else:
            self._transform = transform
