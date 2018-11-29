import numpy as np
from numba import jit


class Canvas:
    def __init__(self, width, height, pixels=None):
        self.width = width
        self.height = height
        self.px = pixels

        if pixels is None:
            self.px = np.zeros((width, height, 3))

    def pixels(self):
        return self.px

    def PPM(self):
        ppm_string = "P3\n{} {}\n255\n".format(self.width, self.height)

        for row_id in range(self.height):
            current_line = ""
            colors = self.px[:, row_id]
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
