import numpy as np
from .config import settings

from collections import defaultdict
from itertools import chain
from operator import methodcaller
import random as rnd

rnd.seed(0)


def random(min_max):
    if type(min_max) is tuple:
        return rnd.uniform(*min_max)
    else:
        return rnd.uniform(0, min_max)

def merge_data(one, two):
    dd = defaultdict(list)

    # iterate dictionary items
    dict_items = map(methodcaller('items'), (one, two))
    for k, v in chain.from_iterable(dict_items):
        dd[k].extend(v)

    return dd

def flatten_data(depth, data, names):
    l = []
    for key1, level1 in data[0].items():
        for key2, level2 in level1.items():
            # I know this should be recursive, fine for now
            if depth == 1:
                for key3, level3 in enumerate(level2):
                    d = {
                        names[0]: key1,
                        names[1]: key2
                    }
                    for i, n in enumerate(names[2]):
                        d[n] = data[i][key1][key2][key3]
                    l.append(d)
            elif depth == 2:
                for key3, level3 in level2.items():
                    for key4, level4 in enumerate(level3):
                        d = {
                            names[0]: key1,
                            names[1]: key2,
                            names[2]: key3
                        }
                        for i, n in enumerate(names[3]):
                            d[n] = data[i][key1][key2][key3][key4]
                        l.append(d)

    return l
