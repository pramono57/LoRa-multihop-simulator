import numpy as np
from .config import settings

import random as rnd
rnd.seed(0)


def random(min_max):
    if type(min_max) is tuple:
        return rnd.uniform(*min_max)
    else:
        return rnd.uniform(0, min_max)
