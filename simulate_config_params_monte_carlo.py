from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys

from multihop.config import settings
from multihop.utils import data_to_df
from multihop.preambles import preambles

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

random.seed(9999)
np.random.seed(9999)

# DF save settings
filename = "results/monte_carlo_measure_aggregation.csv"

# Scenario settings
scenario_settings = [
    {
        "name": "MEASURE_INTERVAL_S",
        "min": 2 * 60,
        "max": 60 * 60,
        "unit": 60
    }, {
        "name": "TX_AGGREGATION_TIMER_NOMINAL",
        "min": 2 * 60,
        "max": 60 * 60,
        "unit": 60
    }
]
number_of_scenarios = 2

each_time = 1  # Run each scenario 10 times
run_time = 60 * 60  # Simulate for 1 day: 60*60*24

# Generate scenarios
scenarios = []
for i in range(0, number_of_scenarios):
    scenario = {}
    for setting in scenario_settings:
        scenario[setting["name"]] = round(np.random.uniform(setting["min"], setting["max"])
                                          / setting["unit"]) * setting["unit"]
    scenarios.append(scenario)

# Generate network
network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)
network.plot_network()

result = pd.DataFrame()

logging.info("Start simulation")

for i_s, scenario in enumerate(scenarios):

    # Set settings from scenario
    for setting_name, setting_value in scenario.items():
        settings.update({setting_name: setting_value})

    logging.info(f"Simulating for  {i_s}")

    random.seed(5555)
    np.random.seed(5555)

    for r in range(0, each_time):
        network = network.copy()
        network.run(run_time)

        df = data_to_df({"pdr": network.hops_statistic("pdr"),
                         "plr": network.hops_statistic("plr"),
                         "aggregation_efficiency": network.hops_statistic("aggregation_efficiency"),
                         "energy": network.hops_statistic("energy")})

        for setting_name, setting_value in scenario.items():
            df[setting_name] = [setting_value] * len(df)

        result = pd.concat([result, df], ignore_index=True)

logging.info("Simulation done")

result.to_csv(filename)
logging.info("Written to file")

print("test")
