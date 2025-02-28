from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys
import copy
import multiprocessing as mp

from multihop.config import settings
from multihop.utils import data_to_df
from multihop.preambles import preambles


def run_helper(args):
    logging.info("Running network")
    _network = Network(settings=args["settings"], map=args["map"])
    _network.run()
    return {
        "settings": _network.settings,

        "pdr": _network.hops_statistic("pdr"),
        "plr": _network.hops_statistic("plr"),
        "aggregation_efficiency": _network.hops_statistic("aggregation_efficiency"),
        "energy": _network.hops_statistic("energy"),
        "latency": _network.hops_statistic("energy")
    }


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    random.seed(9999)
    np.random.seed(9999)

    # DF save settings
    filename = "results/monte_carlo_measure_aggregation_23_02_17.csv"

    # Scenario settings
    scenario_settings = [
        {
            "name": "LORA_SF",
            "min": 7,
            "max": 8,
            "unit": 1
        }
    ]
    number_of_scenarios = 1

    each_time = 1  # Run each scenario 10 times

    # Generate network
    network = Network()
    map = network.get_node_map()

    pool = mp.Pool(math.floor(mp.cpu_count() / 2))

    arg_list = []
    result = pd.DataFrame()

    # Generate settings
    for i in range(0, number_of_scenarios):
        _settings = copy.deepcopy(settings)

        # Loop over all variable settings and change them
        for setting in scenario_settings:
            _settings[setting["name"]] = round(np.random.uniform(setting["min"], setting["max"])
                                               / setting["unit"]) * setting["unit"]

        # Make sure preamble is configured at optimum
        _settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]

        # Do the same simulation a number of times and append to lists
        for r in range(0, each_time):
            arg_list.append({"map": map, "settings": _settings})

    logging.info("Starting simulation pool")
    results = pool.map(func=run_helper, iterable=arg_list)

    for _result in results:
        df = data_to_df({"pdr": _result["pdr"],
                         "plr": _result["plr"],
                         "aggregation_efficiency": _result["aggregation_efficiency"],
                         "energy": _result["energy"],
                         "latency": _result["latency"]})

        for scenario in scenario_settings:
            df[scenario["name"]] = [_result["settings"][scenario["name"]]] * len(df)

        result = pd.concat([result, df], ignore_index=True)

    logging.info("Simulation done")

    result.to_csv(filename)
    logging.info("Written to file")

    print("test")
