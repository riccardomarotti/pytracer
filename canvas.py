import numpy as np


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def pixels(self):
        return np.zeros((self.height, self.width, 3))
