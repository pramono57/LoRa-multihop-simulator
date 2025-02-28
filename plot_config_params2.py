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
monte_carlo = 30
number_of_nodes = 10
shape = "line"
date = "23_10_21_4"
n_hops = 10

#monte_carlo = 10
#number_of_nodes = 32
#shape = "matrix"
#date = "23_10_20_3"
#n_hops = 3

filename = f"results/raw/simulate_{shape}_{stat}_{setting1}_{setting2}_{monte_carlo}_{date}.csv"

export = True
exportFilenameEnergy = f"results/raw/simulate_{shape}_{stat}_{setting1}_{setting2}_{monte_carlo}_{date}_total_energy.tex"
exportFilenameAvgplots = f"results/raw/simulate_{shape}_{stat}_{setting1}_{setting2}_{monte_carlo}_{date}_avg_hops.tex"


df = pd.read_csv(filename)
df2 = df.groupby([setting1, setting2, stat], as_index=False).sum()

def plot(param):
    fig, ax = plt.subplots(num=param)

    pivot_table = df2.pivot_table(index='LORA_TRANSMIT_POWER', columns='LORA_SF', values='energy',
                                 aggfunc='sum')/monte_carlo
    #crosstab = pd.crosstab(df["hops"], [df[setting1], df[setting2]])
    crosstab = df.groupby(['LORA_TRANSMIT_POWER', 'LORA_SF', 'hops']).size().unstack(fill_value=0).reset_index()
    sum = 0
    for x in range(n_hops):
        sum += crosstab[x]*x

    crosstab["avg"] = sum/(number_of_nodes*monte_carlo)
    crosstab["0_32"] =  crosstab[0]/(number_of_nodes*monte_carlo)
    crosstab["2_32"] =  crosstab[2]/(number_of_nodes*monte_carlo)

    ax = pivot_table.plot(kind='line', marker='o', title='Energy Consumption by LORA Parameters')
    ax.set_xlabel('LORA_TRANSMIT_POWER')
    ax.set_ylabel('Sum of Energy Consumption')
    plt.legend(title='LORA_SF')

    if export:
        ax.get_legend().remove()
        tikzplotlib.save(exportFilenameEnergy)

    plt.plot()
    plt.show(block=False)

    #
    fig, ax = plt.subplots(num=param)
    for key, grp in crosstab.groupby(["LORA_SF"]):
        grp.plot(ax=ax, x=setting1, y='avg', label=key)
        #grp.plot(ax=ax, x=setting1, y='0_32', label=key, ls="--")
        #grp.plot(ax=ax, x=setting1, y='2_32', label=key, ls="dashdot")
        ax.set_xlabel('LORA_TRANSMIT_POWER')
        ax.set_ylabel('Hops')

    if export:
        ax.get_legend().remove()
        tikzplotlib.save(exportFilenameAvgplots)

    plt.legend(title='hops avg')
    plt.show(block=False)



    plt.show(block=False)


plot("pdr")
#plot("plr")
#plot("aggregation_efficiency")
plot("energy")
#plot("energy_per_byte")
#plot("energy_tx_per_byte")
#plot("latency")

print("The end")
