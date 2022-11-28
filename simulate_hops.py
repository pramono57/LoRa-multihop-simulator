from multihop.Network import *
import matplotlib.pyplot as plt

from multihop.Network import *
import matplotlib.pyplot as plt
import random

random.seed(5555)
np.random.seed(19680801)

network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)
network.run(60*30)
network.plot_network()
network.plot_states()
print(network.pdr())

network.plot_hops_statistic("pdr")
network.plot_hops_statistic("plr")
network.plot_hops_statistic("aggregation_efficiency")
network.plot_hops_statistic("energy")

print("Test")