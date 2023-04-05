from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys

from multihop.config import settings
from multihop.utils import merge_data
from multihop.preambles import preambles
setting1 = "TX_AGGREGATION_TIMER_RANDOM"
setting2 = "MEASURE_INTERVAL_S"
stat = "children"

df = pd.read_csv(f"results/raw/simulate_matrix_{stat}_{setting2}_{setting1}_23_02_22.csv")

df = df.groupby([setting1, setting2, "children"], as_index=False).mean()


def plot(param):
    fig, ax = plt.subplots(num=param)
    for key, grp in df.groupby([stat]):
        data = grp.groupby(setting1, as_index=False)[param].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
        data.reset_index(inplace=True)

        data["mean_avg"] = data["mean"].rolling(window=5).mean()
        data.plot(ax=ax, x=setting1, y='mean_avg', label=key)
        plt.fill_between(x=setting1, y1='low', y2='high', alpha=0.3, data=data)

    ax.get_legend().remove()
    tikzplotlib.save(f"results/github_final_simulate_matrix_{stat}_{setting1}_{setting2}_23_02_22_{param}.tex")

    plt.show(block=False)


plot("pdr")
plot("plr")
plot("aggregation_efficiency")
plot("energy")
plot("energy_per_byte")
plot("energy_tx_per_byte")
plot("latency")

print("The end")
