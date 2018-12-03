import numpy as np
from numba import jit


class Canvas:
    def __init__(self, width, height, pixels=None):
        self._width = width
        self._height = height
        if pixels is None:
            self._pixels = np.zeros((width, height, 3))
        else:
            self._pixels = pixels

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def pixels(self):
        return self._pixels

    def write_pixel(self, x, y, color):
        self.pixels[x, y] = color

    def pixel_at(self, x, y):
        return self.pixels[x, y]

    def to_PPM(self):
        ppm_string = "P3\n{} {}\n255\n".format(self.width, self.height)

        for row_id in range(self.height):
            current_line = ""
            colors = self.pixels[:, row_id]
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
