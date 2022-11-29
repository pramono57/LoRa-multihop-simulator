from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging

from multihop.config import settings
import multihop.utils

random.seed(5555)
np.random.seed(19680801)
logging.getLogger().setLevel(logging.WARNING)

filename = "results/simulate_funnel_size_aggregation_timer.csv"

setting = "TX_AGGREGATION_TIMER_NOMINAL"
values = range(1*60, 2*60, 2*60)

sizes = range(1, 10, 1)

pdr = {}
plr = {}
aggregation_efficiency = {}
energy = {}

for value in values:
    settings.update({setting: value})

    for size in sizes:
        network = Network(shape="funnel", size_x=100, size_y=0, levels=[1, 1, size])
        network.plot_network()
        network.run(60*30)

        if value not in pdr:
            pdr[value] = {}
            plr[value] = {}
            aggregation_efficiency[value] = {}
            energy[value] = {}

        pdr[value][size] = network.hops_statistic("pdr")
        plr[value][size] = network.hops_statistic("plr")
        aggregation_efficiency[value][size] = network.hops_statistic("aggregation_efficiency")
        energy[value][size] = network.hops_statistic("energy")


df = pd.DataFrame(flatten_data(2,
                               [pdr, plr, aggregation_efficiency, energy],
                               [setting, "size", "hops", ["pdr", "plr", "aggregation_efficiency", "energy"]]))

# Only interested in the middle node
df = df[df.hops == 0]
df.to_csv(filename)

fig, ax = plt.subplots()

for key, grp in df.groupby([setting]):
    data = grp.groupby('size', as_index=False)['energy'].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
    data.reset_index(inplace=True)

    data.plot(ax=ax, x='size', y='mean', label=key)
    #plt.fill_between(x='size', y1='low', y2='high', alpha=0.3, data=data)

plt.show()
test = "test"

plt.show()
