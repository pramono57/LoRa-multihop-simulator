from Nodes import Gateway, SensorNode
from config import settings
import simpy

print(settings.ENERGY_CAD_CYCLE_mJ)

simpy_env = simpy.Environment()
node1 = SensorNode(simpy_env, 1)
node2 = SensorNode(simpy_env, 2)
node3 = SensorNode(simpy_env, 2)

nodes = [node1, node2, node3]

node1.add_nodes(nodes)
node2.add_nodes(nodes)
node3.add_nodes(nodes)

simpy_env.process(node1.run())
# simpy_env.process(node2.run())
# simpy_env.process(node3.run())

simpy_env.run(until=10*60)

node1.plot_states()
node2.plot_states()
node3.plot_states()

