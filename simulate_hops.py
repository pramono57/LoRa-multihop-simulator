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

# network = Network()
# network.plot_network()

# network.run()
# network.save()

network = Network.load("results/2023-03-06 17-20-49_network.dill")
network.plot_network()

#network.plot_hops_statistic("pdr", type="cdf")
# network.plot_hops_statistic("plr")
network.plot_hops_statistic("aggregation_efficiency", type="cdf")
network.plot_hops_statistic("energy_per_byte", type="cdf")
network.plot_hops_statistic("energy_tx_per_byte", type="cdf")
# network.plot_hops_statistic("latency")
# network.plot_hops_statistic_energy_per_state()

network.save_settings()

print("Test")
