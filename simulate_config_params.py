import math

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

settings.PREAMBLE_DURATION_S = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S]

setting1 = "MEASURE_INTERVAL_S"
#values1 = [*range(5,settings.MEASURE_INTERVAL_S+11,math.floor(settings.MEASURE_INTERVAL_S/100))]  # 60*60
values1 = [settings.MEASURE_INTERVAL_S]  # 60*60

setting2 = "NETWORK_DENSITY"
#values2 = [0]
#values2 = [*range(math.floor(settings.MEASURE_INTERVAL_S/50), math.floor(settings.MEASURE_INTERVAL_S)+1, math.floor(settings.MEASURE_INTERVAL_S/100))]
#values2 = [*range(math.floor(settings.TX_AGGREGATION_TIMER_RANDOM[0]/90), math.floor(settings.TX_AGGREGATION_TIMER_RANDOM[0])*3, math.floor(settings.TX_AGGREGATION_TIMER_RANDOM[0]/30))]
#values2 = [*range(math.floor(settings.PREAMBLE_DURATION_S*1000*0), math.floor(settings.PREAMBLE_DURATION_S*1000*0.375), math.floor(settings.PREAMBLE_DURATION_S*1000*0.01))]
#values2 = [x / 1000 for x in values2]
#values2 = [*range(2, 64+1, 2)]
values2 = [300,200,100,50,30,20,10]

stat = "children"

def run_helper(args):
    logging.info("Running network")
    _network = Network(settings=args["settings"], map=args["map"])
    _network.run()
    return {
        "settings": _network.settings,

        "pdr": _network.statistic(stat, "pdr"),
        "plr": _network.statistic(stat, "plr"),
        "aggregation_efficiency": _network.statistic(stat, "aggregation_efficiency"),
        "energy": _network.statistic(stat, "energy"),
        "energy_per_byte": _network.statistic(stat, "energy_per_byte"),
        "energy_tx_per_byte": _network.statistic(stat, "energy_tx_per_byte"),
        "latency": _network.statistic(stat, "latency")
    }


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    monte_carlo = 10  # Run each setting 10 times

    filename = f"results/simulate_matrix_{stat}_{setting1}_{setting2}_23_02_26.csv"

    # Generate first network to get node positions
    random.seed(5555)
    np.random.seed(5555)
    network = Network()
    map = network.get_node_map()

    pdr = {}
    plr = {}
    aggregation_efficiency = {}
    energy = {}
    energy_per_byte = {}
    energy_tx_per_byte = {}
    latency = {}

    pool = mp.Pool(math.floor(0.8  * mp.cpu_count()))

    logging.info("Making list of settings and prepare for data storage")
    arg_list = []
    results = []
    for value2 in values2:
        for value1 in values1:
            # Update what we're looping
            _settings = copy.deepcopy(settings)
            if type(settings[setting1]) is not tuple:
                _settings.update({setting1: value1})
            else:
                _settings.update({setting1: value1})

            if type(settings[setting2]) is not tuple:
                _settings.update({setting2: value2})
            else:
                _settings.update({setting2: (value2,value2)})

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
                energy_per_byte[value2] = {}
                energy_tx_per_byte[value2] = {}
                latency[value2] = {}

    # Go simulation, go!
    results = pool.map(func=run_helper, iterable=arg_list)

    logging.info("Simulation done, now processing results")

    for result in results:
        value2 = result["settings"][setting2]
        if type(value2) is tuple:
            value2 = result["settings"][setting2][0]
        value1 = result["settings"][setting1]
        if type(value1) is tuple:
            value1 = result["settings"][setting1][0]
        if value1 not in pdr[value2]:
            pdr[value2][value1] = result["pdr"]
            plr[value2][value1] = result["plr"]
            aggregation_efficiency[value2][value1] = result["aggregation_efficiency"]
            energy[value2][value1] = result["energy"]
            energy_per_byte[value2][value1] = result["energy_per_byte"]
            energy_tx_per_byte[value2][value1] = result["energy_tx_per_byte"]
            latency[value2][value1] = result["latency"]
        else:
            merge_data(pdr[value2][value1], result["pdr"])
            merge_data(plr[value2][value1], result["plr"])
            merge_data(aggregation_efficiency[value2][value1], result["aggregation_efficiency"])
            merge_data(energy[value2][value1], result["energy"])
            merge_data(energy[value2][value1], result["energy_per_byte"])
            merge_data(energy[value2][value1], result["energy_tx_per_byte"])
            merge_data(latency[value2][value1], result["latency"])

    df = pd.DataFrame(flatten_data(2,
                                   [pdr, plr, aggregation_efficiency, energy, energy_per_byte, energy_tx_per_byte, latency],
                                   [setting2, setting1, stat, ["pdr", "plr", "aggregation_efficiency", "energy", "energy_per_byte", "energy_tx_per_byte", "latency"]]))
    df.to_csv(filename)

    logging.info("Written to file")

    fig, ax = plt.subplots()

    for key, grp in df.groupby([stat]):
        data = grp.groupby(setting1, as_index=False)['energy'].agg({'low': 'min', 'high': 'max', 'mean': 'mean'})
        data.reset_index(inplace=True)

        data.plot(ax=ax, x=setting1, y='mean', label=key)
        plt.fill_between(x=setting1, y1='low', y2='high', alpha=0.3, data=data)

    plt.show()
    print("The end")
