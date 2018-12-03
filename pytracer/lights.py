import numpy as np


class PointLight:
    def __init__(self, position, intensity):
        self._position = position
        self._intensity = intensity

    @property
    def position(self):
        return self._position

    @property
    def intensity(self):
        return self._intensity
