from multihop.Network import *
import matplotlib.pyplot as plt

network = Network(shape = "matrix", size_x = 250, size_y = 250, n_x = 5, n_y = 5)
#network.plot_network()
network.run(60*30)
network.plot_states()
print(network.pdr())

data = {}
for node in network.nodes:
	if node.type == NodeType.SENSOR:
		if node.route is not None:
			hops = node.route.find_best()["hops"]
			if data.get(hops, None) is None:
				data[hops] = [node.pdr()]
			else:
				data[hops].append(node.pdr())

data = dict(sorted(data.items()))

fig = plt.figure()
labels, pltdata = [*zip(*data.items())]  # 'transpose' items to parallel key, value lists
plt.boxplot(pltdata)
plt.xticks(range(1, len(labels) + 1), labels)
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


