from pytracer.transformations import identity_matrix, invert
from pytracer.tuples import point, normalize
from pytracer.rays import Ray
from pytracer.canvas import Canvas
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

    def ray_for_pixel(self, px, py):
        xoffset = (px+0.5) * self.pixel_size
        yoffset = (py+0.5) * self.pixel_size

        world_x = self._half_width - xoffset
        world_y = self._half_height - yoffset

        inverse_camera_transform = invert(self._transform)
        pixel = inverse_camera_transform(point(world_x, world_y, -1))
        origin = inverse_camera_transform(point(0, 0, 0))
        direction = normalize(pixel - origin)

        return Ray(origin, direction)

    def render(self, world):
        image = Canvas(self._hsize, self._vsize)

        for y in range(self._vsize):
            for x in range(self._hsize):
                ray = self.ray_for_pixel(x, y)
                color = world.color_at(ray)
                image.write_pixel(x, y, color)

        return image
