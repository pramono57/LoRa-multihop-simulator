from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys

from multihop.config import settings
from multihop.utils import merge_data
from multihop.preambles import preambles

setting1 = "MEASURE_INTERVAL_S"
setting2 = "TX_AGGREGATION_TIMER_NOMINAL"

df = pd.read_csv(f"results/simulate_matrix_{setting1}_{setting2}.csv")

fig, ax = plt.subplots()

for key, grp in df.groupby(['hops']):
    data = grp.groupby(setting2, as_index=False)['aggregation_efficiency'].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
    data.reset_index(inplace=True)

    data.plot(ax=ax, x=setting2, y='mean', label=key)
    plt.fill_between(x=setting2, y1='low', y2='high', alpha=0.3, data=data)

plt.show()
print("The end")