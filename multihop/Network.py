from .utils import *
from .config import settings
from .Timers import TxTimer, TimerType
from .Packets import Message, MessageType
from .Links import LinkTable
from .Routes import Route
from .Positions import Position
from .Nodes import Node, NodeType, GatewayState, NodeState

import numpy as np
import simpy
import math
from operator import methodcaller

class Network:
    def __init__(self, **kwargs):
        self.nodes = []
        self.simpy_env = simpy.Environment()
        self.link_table = None

        n_x = None
        n_y = None
        n = None

        positioning = kwargs.get('shape', None)
        size_x = kwargs.get('size_x', None)
        size_y = kwargs.get('size_y', None)

        density = kwargs.get('density', None)  # Density: how many nodes per km2
        if density is not None:
            n_x = round(math.sqrt(density)/1000*size_x)
            n_y = round(math.sqrt(density) / 1000 * size_y)

        else:
            n_x = kwargs.get('n_x', None)
            n_y = kwargs.get('n_y', None)
            if n_x is None or n_y is None:
                n = kwargs.get('n', None)
            else:
                number_of_nodes = n_x * n_y

        rnd = kwargs.get("size_random")
        if rnd is None:
            rnd = 0

        g_x = kwargs.get('g_x', None)
        g_y = kwargs.get('g_y', None)
        if g_x is None:
            g_x = 0
        if g_y is None:
            g_y = 0

        self.nodes.append(Node(self.simpy_env, 0, Position(g_x, g_y), NodeType.GATEWAY))

        if positioning == "random":
            for x in range(1, n+1):
                self.nodes.append(Node(self.simpy_env, x, Position(*np.random.uniform(-size_x/2, size_y/2, size=2)), NodeType.SENSOR))

        elif positioning == "line":
            if n == 1:
                self.nodes.append(Node(self.simpy_env, 1, Position(size_x, size_y), NodeType.SENSOR))
            else:
                for x in range(1, n+1):
                    self.nodes.append(Node(self.simpy_env, x, Position(-size_x/2+(x-1)*size_x/n, -size_y/2+(x-1)*size_y/n), NodeType.SENSOR))

        elif positioning == "matrix":
            uid = 1
            for y in range(0, n_y):
                for x in range(0, n_x):
                    self.nodes.append(Node(self.simpy_env, uid,
                                           Position(-size_x / 2 + x * size_x / (n_x - 1) + np.random.uniform(-rnd/2, rnd),
                                                    -size_y / 2 + y * size_y / (n_y - 1) + np.random.uniform(-rnd/2, rnd)),
                                           NodeType.SENSOR))
                    uid += 1

        recalc = self.evaluate_distances()
        while recalc:
            recalc = self.evaluate_distances()

        self.link_table = LinkTable(self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.nodes, self.link_table)

    def evaluate_distances(self):
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1.uid is not node2.uid and node1.uid < node2.uid:
                        d = np.sqrt(np.abs(node1.position.x - node2.position.x)**2 +
                                    np.abs(node1.position.y - node2.position.y)**2)
                        if d < 1:
                            # Distance between two nodes is too small: move each node half of what is needed
                            t = -(1-d)/2
                            x1 = (1 - t/d) * node1.position.x + t / d * node2.position.x
                            y1 = (1 - t/d) * node1.position.y + t / d * node2.position.y
                            t = d+(1-d)/2
                            x2 = (1 - t/d) * node1.position.x + t / d * node2.position.x
                            y2 = (1 - t/d) * node1.position.y + t / d * node2.position.y

                            node1.position = Position(x1, y1)
                            node2.position = Position(x2, y2)

                            return True


    def run(self, time):
        # First get max hop count in total network to establish how long the simulation should be extended
        paths = self.link_table.get_all_pairs_shortest_path()
        max_hops = 0
        for path in paths:
            hops = len(max(path[1].values(), key=len))
            if hops > max_hops:
                max_hops = hops

        # Extend simulation time
        simulation_time = time + (max_hops * (settings.TX_AGGREGATION_TIMER_MAX
                                              + settings.TX_AGGREGATION_TIMER_RANDOM[1]
                                              + settings.TX_COLLISION_TIMER_NOMINAL
                                              + settings.TX_COLLISION_TIMER_RANDOM[1]))

        # Start all nodes
        for node in self.nodes:
            node.sense_until = time
            self.simpy_env.process(node.run())

        self.simpy_env.run(until=simulation_time)

    def plot_network(self):
        self.link_table.plot()

    def plot_states(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(self.nodes), sharex=True, sharey=True)
        for i, node in enumerate(self.nodes):
            node.plot_states(ax[i])

        ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6], ["INIT", "ZZZ", "CAD", "RX", "SNS", "P_TX", "TX"])
        plt.show(block=False)

    def hops_statistic(self, stat):
        method = methodcaller(stat)

        data = {}
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                if node.route is not None:
                    hops = node.route.find_best()["hops"]
                    if data.get(hops, None) is None:
                        data[hops] = [method(node)]
                    else:
                        data[hops].append(method(node))

        data = dict(sorted(data.items()))

        return data

    def plot_hops_statistic(self, stat):
        import matplotlib.pyplot as plt

        data = self.hops_statistic(stat)

        fig = plt.figure()
        labels, pltdata = [*zip(*data.items())]  # 'transpose' items to parallel key, value lists
        plt.boxplot(pltdata)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.show(block=False)

    def pdr(self):
        payloads_sent = 0
        payloads_received = 0
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                payloads_sent += len(node.own_payloads_sent)
                payloads_received += len(node.own_payloads_arrived_at_gateway)

        return payloads_received/payloads_sent


