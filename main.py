from matplotlib import pyplot as plt

from Nodes import Node, NodeType, Position
from Links import LinkTable
from config import settings
import simpy

simpy_env = simpy.Environment()

nodes = []

nodes.append(Node(simpy_env, 0, Position(0,0), NodeType.GATEWAY))
number_of_nodes = 10
for x in range(1, number_of_nodes+1):
    nodes.append(Node(simpy_env, x, Position((x%2)*10, x*30), NodeType.SENSOR))

link_table = LinkTable(nodes)
link_table.plot()

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

# fig, ax = plt.subplots(number_of_nodes+1, sharex=True, sharey=True)
# for i, node in enumerate(nodes):
#     node.plot_states(ax[i])

# ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6], ["INIT", "ZZZ", "CAD", "RX", "SNS", "P_TX", "TX"])
# plt.show()

