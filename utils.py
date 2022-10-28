import numpy as np
from config import settings

import random as rnd
rnd.seed(0)


def random(min_max):

    if type(min_max) is tuple:
        return rnd.uniform(*min_max)
    else:
        return rnd.uniform(0, min_max)


def get_rss(n1, n2):
    d = distance(n1.position, n2.position)
    return settings.tp - path_loss(d)


def distance(p1, p2):
    return np.sqrt(np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y))


def path_loss(d):
    return 74.85 + 2.75 * 10 * np.log10(d) # currently no random shadowing +


def snr(rss):
    return rss + 116.86714407 # thermal noise for 25°C 500kHz BW

def in_range(n1, n2):
    return get_rss(n1, n2) > settings.sensitivity
