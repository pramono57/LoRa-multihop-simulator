from multihop.Network import *
import matplotlib.pyplot as plt
import random

random.seed(5555)
np.random.seed(19680801)

network = Network(shape="matrix-random", size_x=240, size_y=240, n_x=6, n_y=6, size_random=10)
network.run(60*30)
network.plot_network()
network.plot_states()
print(network.pdr())

data_pdr = {}
data_plr = {}
data_aggregation_efficiency = {}

for node in network.nodes:
	if node.type == NodeType.SENSOR:
		if node.route is not None:
			hops = node.route.find_best()["hops"]
			if data_pdr.get(hops, None) is None:
				data_pdr[hops] = [node.pdr()]
				data_plr[hops] = [node.plr()]
				data_aggregation_efficiency[hops] = [node.aggregation_efficiency()]
			else:
				data_pdr[hops].append(node.pdr())
				data_plr[hops].append(node.plr())
				data_aggregation_efficiency[hops].append(node.aggregation_efficiency())

data_pdr = dict(sorted(data_pdr.items()))
data_plr = dict(sorted(data_plr.items()))

fig = plt.figure()
labels, pltdata_pdr = [*zip(*data_pdr.items())]  # 'transpose' items to parallel key, value lists
plt.boxplot(pltdata_pdr)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show(block=False)

fig = plt.figure()
labels, pltdata_plr = [*zip(*data_plr.items())]  # 'transpose' items to parallel key, value lists
plt.boxplot(pltdata_plr)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show(block=False)

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


