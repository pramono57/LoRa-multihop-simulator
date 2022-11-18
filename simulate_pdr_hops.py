from multihop.Network import *
import matplotlib.pyplot as plt

network = Network(shape="matrix", size_x=250, size_y=30, n_x=5, n_y=2)
network.plot_network()

network.run(60*30)
print(network.pdr())

data = {}
for node in network.nodes:
	if node.type == NodeType.SENSOR:
		if node.route.find_best() is not None:
			hops = node.route.find_best()["hops"]
			if data.get(hops, None) is not None:
				data[hops] = [node.pdr()]
			else:
				data[hops].append(node.pdr())

data = dict(sorted(data.items()))

fig = plt.figure()
labels, plt_data = [*zip(*data.items())]  # 'transpose' items to parallel key, value lists
plt.boxplot(plt_data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()