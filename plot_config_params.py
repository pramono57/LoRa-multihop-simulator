from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys

from multihop.config import settings
from multihop.utils import merge_data
from multihop.preambles import preambles
setting1 = "LORA_TRANSMIT_POWER"
setting2 = "LORA_SF"
stat = "hops"
monte_carlo = 10

df = pd.read_csv(f"results/raw/simulate_matrix_{stat}_{setting1}_{setting2}_{monte_carlo}_23_10_15.csv")

df = df.groupby([setting1, setting2, stat], as_index=False).mean()


def plot(param):
    fig, ax = plt.subplots(num=param)
    for key, grp in df.groupby([stat]):
        data = grp.groupby(setting2, as_index=False)[param].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
        data.reset_index(inplace=True)

        data["mean_avg"] = data["mean"].rolling(window=1).mean()
        data.plot(ax=ax, x=setting2, y='mean_avg', label=key)
        plt.title(param)
        plt.fill_between(x=setting2, y1='low', y2='high', alpha=0.3, data=data)

    #ax.get_legend().remove()
    #tikzplotlib.save(f"results/raw/github_final_simulate_matrix_{stat}_{setting1}_{setting2}_23_02_22_{param}.tex")

    plt.show(block=False)


plot("pdr")
plot("plr")
plot("aggregation_efficiency")
plot("energy")
plot("energy_per_byte")
plot("energy_tx_per_byte")
plot("latency")

print("The end")
