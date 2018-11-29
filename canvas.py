import numpy as np


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
            colors = self.px[:, row_id]
            colors = colors.flatten()
            clamped_colors = clamp(colors)
            ppm_string += " ".join(clamped_colors.astype(str))
            ppm_string += "\n"

        return ppm_string


@np.vectorize
def clamp(color):
    return int(max(0, min(255, color * 255)))
