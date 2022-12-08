from multihop.Network import *
import matplotlib.pyplot as plt

from multihop.Network import *
import matplotlib.pyplot as plt
import random
import logging
from multihop.preambles import preambles

logging.getLogger().setLevel(logging.INFO)

random.seed(5555)
np.random.seed(19680801)

settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]


network = Network(shape="matrix",  size_x=180, size_y=120, n_x=4, n_y=4, size_random=3)
network.plot_network()

network.run()
network.plot_network()

network.plot_hops_statistic("pdr")
network.plot_hops_statistic("plr")
network.plot_hops_statistic("aggregation_efficiency")
network.plot_hops_statistic("energy", relative="min")
network.plot_hops_statistic("latency")
network.plot_hops_statistic_energy_per_state()

print("Test")
