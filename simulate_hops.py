from multihop.Network import *
import matplotlib.pyplot as plt

from multihop.Network import *
import matplotlib.pyplot as plt
import random
import logging

logging.getLogger().setLevel(logging.INFO)

# TODO: CDF plot and concurrent processing!

random.seed(5555)
np.random.seed(19680801)

network = Network(shape="matrix",  size_x=180, size_y=120, n_x=4, n_y=4, size_random=3)
network.plot_network()

network.run(60*60*24)
network.plot_network()

network.plot_hops_statistic("pdr")
network.plot_hops_statistic("plr")
network.plot_hops_statistic("aggregation_efficiency")
network.plot_hops_statistic("energy")
network.plot_hops_statistic("latency")

print("Test")