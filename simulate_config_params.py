from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd
import logging
import sys
import copy
import multiprocessing as mp

from multihop.config import settings
from multihop.utils import merge_data
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

    monte_carlo = 1  # Run each setting 10 times

    setting1 = "MEASURE_INTERVAL_S"
    values1 = [settings.MEASURE_INTERVAL_S]  # 60*60

    setting2 = "TX_COLLISION_TIMER_NOMINAL"
    values2 = range(1, 100, 10)

    filename = f"results/simulate_matrix_{setting1}_{setting2}.csv"

    # Generate first network to get node positions
    random.seed(5555)
    np.random.seed(5555)
    network = Network(settings=settings, shape="matrix", size_x=180, size_y=120, n_x=4, n_y=4, size_random=3)
    map = network.get_node_map()

    pdr = {}
    plr = {}
    aggregation_efficiency = {}
    energy = {}
    latency = {}

    pool = mp.Pool(math.floor(mp.cpu_count() / 3))

    logging.info("Making list of settings and prepare for data storage")
    arg_list = []
    results = []
    for value2 in values2:
        for value1 in values1:
            # Update what we're looping
            _settings = copy.deepcopy(settings)
            _settings.update({setting1: value1})
            _settings.update({setting2: value2})

            # Make sure preamble is configured at optimum
            _settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]

            # Do the same simulation a number of times and append to lists
            for r in range(0, monte_carlo):
                arg_list.append({"map": map, "settings": _settings})

            # Prepare lists and structs for data storage
            if value2 not in pdr:
                pdr[value2] = {}
                plr[value2] = {}
                aggregation_efficiency[value2] = {}
                energy[value2] = {}
                latency[value2] = {}

    # Go simulation, go!
    results = pool.map(func=run_helper, iterable=arg_list)

    logging.info("Simulation done, now processing results")

    for result in results:
        value2 = result["settings"][setting2]
        value1 = result["settings"][setting1]
        if value1 not in pdr[value2]:
            pdr[value2][value1] = result["pdr"]
            plr[value2][value1] = result["plr"]
            aggregation_efficiency[value2][value1] = result["aggregation_efficiency"]
            energy[value2][value1] = result["energy"]
            latency[value2][value1] = result["latency"]
        else:
            merge_data(pdr[value2][value1], result["pdr"])
            merge_data(plr[value2][value1], result["plr"])
            merge_data(aggregation_efficiency[value2][value1], result["aggregation_efficiency"])
            merge_data(energy[value2][value1], result["energy"])
            merge_data(latency[value2][value1], result["latency"])

    df = pd.DataFrame(flatten_data(2,
                                   [pdr, plr, aggregation_efficiency, energy],
                                   [setting2, setting1, "hops", ["pdr", "plr", "aggregation_efficiency", "energy"]]))
    df.to_csv(filename)

    logging.info("Written to file")

    fig, ax = plt.subplots()

    for key, grp in df.groupby(['hops']):
        data = grp.groupby(setting1, as_index=False)['energy'].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
        data.reset_index(inplace=True)

        data.plot(ax=ax, x=setting1, y='mean', label=key)
        plt.fill_between(x=setting1, y1='low', y2='high', alpha=0.3, data=data)

    plt.show()
    print("The end")
