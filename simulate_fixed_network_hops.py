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

network_route = {0: {
    21: {},
    20: {
        4: {
            7: {},
            8: {},
            9: {
                18: {},
                13: {
                    15: {},
                    26: {
                        27: {}
                    }
                },
                14: {}
            },
            10: {}
        },
        5: {},
        6: {
            11: {}
        }
    },
    3: {}}}

print(flatten_node_tree(network_route))

network = Network(shape="random", size_x=30, size_y=30, fixed_route=network_route)
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
