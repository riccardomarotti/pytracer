

class World:
    def __init__(self):
        self._objects = []
        self._light = None

    @property
    def light(self):
        return self._light

    def __len__(self):
        return len(self._objects)
