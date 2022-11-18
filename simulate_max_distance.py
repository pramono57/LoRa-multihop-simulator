# Simulate max distance between two nodes

from multihop.Network import *
from multihop.config import settings

network = Network(n = 1, shape = "line", size_x = 30, size_y = 0)

positions = []

if settings.SHADOWING_ENABLED is False:
	times = 1
	print("No shadowing enabled, only running once.")
else:
	times = 1000
	print(f"Shadowing enabled, running {times} times.")

for i in range(0, times):
	while network.nodes[1].link_table.get_from_uid(0, 1).in_range():
	    network.nodes[1].position.x += 0.1

	positions.append(network.nodes[1].position.x)

position = sum(positions)/len(positions)

print(f"Connection lost at average {position}m.")

