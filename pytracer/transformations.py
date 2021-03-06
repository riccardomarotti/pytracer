import numpy as np
import math
from pytracer.tuples import normalize


def identity_matrix(x=None):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def concat(t1, *tn):
    if len(tn) == 0:
        return t1

    t2 = tn[0]
    rest = tn[1:]
    return concat(lambda p: t1(t2(p)), *rest)


def translation(x, y, z):
    T = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def scaling(x, y, z):
    T = np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def rotation_x(alpha):
    T = np.array([
        [1, 0, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha), 0],
        [0, math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def rotation_y(alpha):
    T = np.array([
        [math.cos(alpha), 0, math.sin(alpha), 0],
        [0, 1, 0, 0],
        [-math.sin(alpha), 0, math.cos(alpha), 0],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def rotation_z(alpha):
    T = np.array([
        [math.cos(alpha), -math.sin(alpha), 0, 0],
        [math.sin(alpha), math.cos(alpha), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def shearing(xy, xz, yx, yz, zx, zy):
    T = np.array([
        [1, xy, xz, 0],
        [yx, 1, yz, 0],
        [zx, zy, 1, 0],
        [0, 0, 0, 1]
    ])
    return lambda p: T.dot(p)


def invert(t):
    T = t(identity_matrix())
    Tinv = np.linalg.inv(T)
    return lambda p: Tinv.dot(p)


def transpose(t):
    T = t(identity_matrix())
    Tt = np.transpose(T)
    return lambda p: Tt.dot(p)


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
    t = translation(-from_[0], -from_[1], -from_[2])(identity_matrix())

    return lambda p: orientation.dot(t).dot(p)
