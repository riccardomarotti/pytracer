from pytracer.spheres import Sphere
from pytracer.transformations import scaling, rotation_y, rotation_x, translation, view_transformation
from pytracer.materials import Material
from pytracer.colors import color
from pytracer.camera import Camera
from pytracer.tuples import point, vector
from pytracer.lights import PointLight
from pytracer.world import World
import math

floor_material = Material(color=color(1, 0.9, 0.9), specular=0)
floor = Sphere(transformation=scaling(10, 0.01, 10), material=floor_material)

left_wall_transformation = translation(0, 0, 5).dot(
    rotation_y(-math.pi/4)).dot(rotation_x(math.pi/2)).dot(scaling(10, 0.01, 10))
left_wall = Sphere(transformation=left_wall_transformation,
                   material=floor_material)

right_wall_transform = translation(0, 0, 5).dot(rotation_y(
    math.pi/4)).dot(rotation_x(math.pi/2)).dot(scaling(10, 0.01, 10))
right_wall = Sphere(transformation=right_wall_transform,
                    material=floor_material)

middle_transform = translation(-0.5, 1, 0.5)
middle_material = Material(color=color(0.1, 1, 0.5), diffuse=0.7, specular=0.3)
middle = Sphere(material=middle_material, transformation=middle_transform)

right_transform = translation(1.5, 0.5, -0.5).dot(scaling(0.5, 0.5, 0.5))
right_material = Material(color=color(0.5, 1, 0.1), diffuse=0.7, specular=0.3)
right = Sphere(material=right_material, transformation=right_transform)

left_transform = translation(-1.5, 0.33, -0.75).dot(scaling(0.33, 0.33, 0.33))
lef_material = Material(color=color(1, 0.8, 0.1), diffuse=0.7, specular=0.3)
left = Sphere(material=lef_material, transformation=left_transform)

camera_transform = view_transformation(
    point(0, 1.5, -5), point(0, 1, 0), vector(0, 1, 0))
camera = Camera(1000, 500, math.pi/3, transform=camera_transform)

light = PointLight(point(-10, 10, -10), color(1, 1, 1))
world = World(light, floor, left_wall, right_wall, middle, left, right)

print(camera.render(world).to_PPM())
