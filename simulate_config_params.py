from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd

from multihop.config import settings
import multihop.utils

random.seed(5555)
np.random.seed(19680801)

setting = "MEASURE_INTERVAL_S"
values = range(2*60, 20*60, 2*60)

network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)
network.plot_network()

pdr = {}
plr = {}
aggregation_efficiency = {}
energy = {}

for value in values:
    settings.update({setting: value})

    network = network.copy()
    network.run(60*30)

    pdr[value] = network.hops_statistic("pdr")
    plr[value] = network.hops_statistic("plr")
    aggregation_efficiency[value] = network.hops_statistic("aggregation_efficiency")
    energy[value] = network.hops_statistic("energy")

df = pd.DataFrame(flatten_data(1,
                               [pdr, plr, aggregation_efficiency, energy],
                               [setting, "hops", ["pdr", "plr", "aggregation_efficiency", "energy"]]))
fig, ax = plt.subplots()

for key, grp in df.groupby(['hops']):
    data = grp.groupby(setting, as_index=False)['energy'].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
    data.reset_index(inplace=True)

    data.plot(ax=ax, x=setting, y='mean', label=key)
    plt.fill_between(x=setting, y1='low', y2='high', alpha=0.3, data=data)

plt.show()
print("test")