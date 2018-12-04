import numpy as np
import math
from pytracer.tuples import normalize


def identity_matrix():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def translation(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])


def scaling(x, y, z):
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])


def rotation_x(alpha):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha), 0],
        [0, math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 0, 1]
    ])


def rotation_y(alpha):
    return np.array([
        [math.cos(alpha), 0, math.sin(alpha), 0],
        [0, 1, 0, 0],
        [-math.sin(alpha), 0, math.cos(alpha), 0],
        [0, 0, 0, 1]
    ])


def rotation_z(alpha):
    return np.array([
        [math.cos(alpha), -math.sin(alpha), 0, 0],
        [math.sin(alpha), math.cos(alpha), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def shearing(xy, xz, yx, yz, zx, zy):
    return np.array([
        [1, xy, xz, 0],
        [yx, 1, yz, 0],
        [zx, zy, 1, 0],
        [0, 0, 0, 1]
    ])


def invert(t):
    return np.linalg.inv(t)


def transpose(t):
    return np.transpose(t)


def view_transformation(from_, to, up):
    forward = normalize(to - from_)[:3]
    left = np.cross(forward, normalize(up)[:3])
    true_up = np.cross(left, forward)

    orientation = np.array([
        [left[0], left[1], left[2], 0],
        [true_up[0], true_up[1], true_up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])
    t = translation(-from_[0], -from_[1], -from_[2])

    return orientation.dot(t)
