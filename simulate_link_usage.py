from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys

from multihop.config import settings
from multihop.utils import merge_data
from multihop.preambles import preambles

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

run_time = 60 * 60  # Simulate for 1 day: 60*60*24

random.seed(5555)
np.random.seed(5555)
network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)

logging.info("Simulation started")

network.run(run_time)
network.plot_network()
network.plot_network_usage()

logging.info("Simulation done")

plt.show()

print("The end")
