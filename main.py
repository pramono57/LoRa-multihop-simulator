from multihop.Network import *
import matplotlib.pyplot as plt
import random
import pandas as pd

from multihop.config import settings

random.seed(5555)
np.random.seed(19680801)

setting = "MEASURE_INTERVAL_S"
values = range(1*60, 60*60, 120)

network = Network(shape="matrix", size_x=200, size_y=200, density=1000, size_random=10)

for value in values:
    settings.update({setting: value})

    network.run(60*30)

    # Todo: process data into pandas to create fancy graphs
    network.hops_statistic("pdr")
    network.hops_statistic("plr")
    network.hops_statistic("aggregation_efficiency")
    network.hops_statistic("energy")

test = "test"

plt.show()

# simpy_env = simpy.Environment()

# nodes = []

# nodes.append(Node(simpy_env, 0, Position(0, 0), NodeType.GATEWAY))
# nodes.append(Node(simpy_env, 1, Position(2, 0), NodeType.SENSOR))

# link_table = LinkTable(nodes)

# for node in nodes:
#     if type(node) is Node:
#         node.add_meta(nodes, link_table)

# while nodes[1].link_table.get_from_uid(0, 1).in_range():
#     nodes[1].position.x += 1

# print("connection lost at")
# print(nodes[1].position.x)
# print("with rss")
# print(nodes[1].link_table.get_from_uid(0, 1).rss())

# number_of_nodes = 10
# for x in range(1, number_of_nodes+1):
#     nodes.append(Node(simpy_env, x, Position((x%2)*10, x*30), NodeType.SENSOR))

# link_table = LinkTable(nodes)
# link_table.plot()

# for node in nodes:
#     if type(node) is Node:
#         node.add_meta(nodes, link_table)

# for node in nodes:
#     simpy_env.process(node.run())

# simpy_env.run(until=32 * 60)

# print("Gateway got these messages:")
# for message in nodes[0].messages_for_me:
#     print(message)

# print("10 sent these payloads that arrived at gateway: with latencies")
# for pl in nodes[10].own_payloads_arrived_at_gateway:
#     print(pl)
#     latency = pl.arrived_at - pl.sent_at
#     print(latency)

# print("8 own data and forwarded:")
# print(nodes[8].message_counter_own_data_and_forwarded_data)
# print("only own data")
# print(nodes[8].message_counter_only_own_data)
# print("only forwarded")
# print(nodes[8].message_counter_only_forwarded_data)


