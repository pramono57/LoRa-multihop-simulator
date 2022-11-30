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

random.seed(9999)
np.random.seed(9999)

# Scenario settings
scenario_settings = [
    {
        "name": "MEASURE_INTERVAL_S",
        "min": 2 * 60,
        "max": 60 * 60
    }, {
        "name": "TX_AGGREGATION_TIMER_NOMINAL",
        "min": 2 * 60,
        "max": 60 * 60
    }
]
number_of_scenarios = 1000

each_time = 10  # Run each scenario 10 times

# Generate scenarios
scenarios = []
for i in range(0, number_of_scenarios):
    scenario = {}
    for setting in scenario_settings:
        scenario[setting["name"]] = np.random.uniform(setting["min"], setting["max"])
    scenarios.append(scenario)

setting1 = "MEASURE_INTERVAL_S"
values1 = range(2 * 60, 10 * 60, 5 * 60)  # 60*60

setting2 = "TX_AGGREGATION_TIMER_NOMINAL"
values2 = range(2 * 60, 10 * 60, 5 * 60)

filename = f"results/simulate_matrix_{setting1}_{setting2}.csv"

run_time = 60 * 60  # Simulate for 1 day: 60*60*24

random.seed(5555)
np.random.seed(5555)
network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)
network.plot_network()

pdr = {}
plr = {}
aggregation_efficiency = {}
energy = {}

logging.info("Start simulation")

for value2 in values2:
    for value1 in values1:
        logging.info(f"Simulating for value2 {value2}")

        random.seed(5555)
        np.random.seed(5555)
        settings.update({setting1: value1})

        settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]

        if value2 not in pdr:
            logging.info(f"\t Simulating for value1 {value1}")

            pdr[value2] = {}
            plr[value2] = {}
            aggregation_efficiency[value2] = {}
            energy[value2] = {}

        for r in range(0, monte_carlo):
            network = network.copy()
            network.run(run_time)

            if value1 not in pdr[value2]:
                pdr[value2][value1] = network.hops_statistic("pdr")
                plr[value2][value1] = network.hops_statistic("plr")
                aggregation_efficiency[value2][value1] = network.hops_statistic("aggregation_efficiency")
                energy[value2][value1] = network.hops_statistic("energy")
            else:
                merge_data(pdr[value2][value1], network.hops_statistic("pdr"))
                merge_data(plr[value2][value1], network.hops_statistic("plr"))
                merge_data(aggregation_efficiency[value2][value1], network.hops_statistic("aggregation_efficiency"))
                merge_data(energy[value2][value1], network.hops_statistic("energy"))

logging.info("Simulation done")

df = pd.DataFrame(flatten_data(2,
                               [pdr, plr, aggregation_efficiency, energy],
                               [setting2, setting1, "hops", ["pdr", "plr", "aggregation_efficiency", "energy"]]))
df.to_csv(filename)

logging.info("Written to file")

print("test")
