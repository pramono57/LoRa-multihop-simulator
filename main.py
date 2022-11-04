from matplotlib import pyplot as plt

from Nodes import Gateway, SensorNode
from config import settings
import simpy

print(settings.POWER_CAD_CYCLE_mW)
print(settings.POWER_SENSE_mW)

simpy_env = simpy.Environment()
gw = Gateway(simpy_env, 0)
node1 = SensorNode(simpy_env, 1)
node2 = SensorNode(simpy_env, 2)
node3 = SensorNode(simpy_env, 3)

nodes = [gw, node1, node2, node3]

node1.add_nodes(nodes)
node2.add_nodes(nodes)
node3.add_nodes(nodes)

simpy_env.process(gw.run())
simpy_env.process(node1.run())
simpy_env.process(node2.run())
simpy_env.process(node3.run())

simpy_env.run(until=10 * 40)

fig, ax = plt.subplots(4, sharex=True, sharey=True)
node1.plot_states(ax[0], plot_labels=False)
node2.plot_states(ax[1], plot_labels=False)
node3.plot_states(ax[2])
gw.plot_states(ax[3])
plt.show()
