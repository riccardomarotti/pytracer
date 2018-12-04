import math
import numpy as np
from pytracer.camera import Camera
from pytracer.transformations import identity_matrix, rotation_y, translation, view_transformation
from pytracer.tuples import point, vector
from pytracer.world import default_world
from pytracer.colors import color


def test_constructing_a_camera():
    hsize = 160
    vsize = 120
    field_of_view = math.pi/2

    c = Camera(hsize, vsize, field_of_view)

    assert(c._hsize == 160)
    assert(c._vsize == 120)
    assert(c._field_of_view == math.pi/2)
    assert(np.array_equal(c._transform, identity_matrix()))


def test_the_pixel_size_for_a_horizontal_canvas():
    c = Camera(200, 125, math.pi/2)
    assert(math.isclose(c.pixel_size, 0.01))


def test_the_pixel_size_for_a_vertical_canvas():
    c = Camera(125, 200, math.pi/2)
    assert(math.isclose(c.pixel_size, 0.01))


def test_constructing_a_ray_through_the_center_of_the_canvas():
    c = Camera(201, 101, math.pi/2)
    r = c.ray_for_pixel(100, 50)

    assert(np.array_equal(point(0, 0, 0), r.origin))
    assert(np.allclose(vector(0, 0, -1), r.direction))


def test_constructing_a_ray_through_the_corner_of_the_canvas():
    c = Camera(201, 101, math.pi/2)
    r = c.ray_for_pixel(0, 0)

    assert(np.array_equal(point(0, 0, 0), r.origin))
    assert(np.allclose(vector(0.66519, 0.33259, -0.66851), r.direction))


def test_constructing_a_ray_when_the_camera_is_transformed():
    c = Camera(201, 101, math.pi/2,
               transform=rotation_y(math.pi/4).dot(translation(0, -2, 5)))
    r = c.ray_for_pixel(100, 50)

    assert(np.allclose(point(0, 2, -5), r.origin))
    assert(np.allclose(vector(math.sqrt(2)/2, 0, -math.sqrt(2)/2), r.direction))


def test_rendering_a_world_with_a_camera():
    w = default_world()
    from_ = point(0, 0, -5)
    to = point(0, 0, 0)
    up = vector(0, 1, 0)
    c = Camera(11, 11, math.pi/2, transform=view_transformation(from_, to, up))

    image = c.render(w)

    assert(np.allclose(image.pixel_at(5, 5), color(
        0.38066, 0.47583, 0.2855), atol=0.00001))
