import numpy as np
from numba import jit


def PPM(width, height, pixels=None):
    if pixels is None:
        pixels = np.zeros((width, height, 3))

    ppm_string = "P3\n{} {}\n255\n".format(width, height)

    for row_id in range(height):
        current_line = ""
        colors = pixels[:, row_id]
        colors = colors.flatten()
        clamped_colors = clamp(colors)
        current_line = " ".join(clamped_colors.astype(str))
        ppm_string += truncate(current_line) + "\n"

    return ppm_string


@np.vectorize
def clamp(color):
    return int(max(0, min(255, color * 255)))


def truncate(s):
    MAX = 70
    x = len(s)
    if(x) < MAX:
        return s

    last_space_index = s[:MAX+1].rfind(' ')
    if last_space_index != -1:
        return s[:last_space_index] + "\n" + s[last_space_index+1:]
