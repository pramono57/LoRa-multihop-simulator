import numpy as np


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def random(cls, size):
        return Position(*np.random.uniform(-size/2, size/2, size=2))
