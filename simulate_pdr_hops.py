from lib.Network import *
import matplotlib.pyplot as plt

network = Network(shape = "line", size_x = 250, size_y = 0, n = 5)
network.plot_network()
network.run(60*60*4)
print(network.pdr())

data = {}
for node in network.nodes:
	if node.type == NodeType.SENSOR:
		if node.route.find_best() != None:
			hops = node.route.find_best()["hops"]
			if data.get(hops, None) == None:
				data[hops] = [node.pdr()]
			else:
				data[hops].append(node.pdr())

data = dict(sorted(data.items()))

fig = plt.figure()
labels, pltdata = [*zip(*data.items())]  # 'transpose' items to parallel key, value lists
plt.boxplot(pltdata)
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()