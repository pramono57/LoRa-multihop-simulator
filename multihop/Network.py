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


class Network:
    def __init__(self, **kwargs):
        self.nodes = []
        self.simpy_env = simpy.Environment()
        self.link_table = None

        n_x = kwargs.get('n_x', None)
        n_y = kwargs.get('n_y', None)
        if n_x is None or n_y is None:
            number_of_nodes = kwargs.get('n', None)
        else: 
            number_of_nodes = n_x * n_y
        positioning = kwargs.get('shape', None)
        size_x = kwargs.get('size_x', None)
        size_y = kwargs.get('size_y', None)

        self.nodes.append(Node(self.simpy_env, 0, Position(0, 0), NodeType.GATEWAY))

        if positioning == "random":
            for x in range(1, number_of_nodes+1):
                self.nodes.append(Node(self.simpy_env, x, Position(*np.random.uniform(-size_x/2, size_y/2, size=2)), NodeType.SENSOR))

        elif positioning == "line":
            if number_of_nodes == 1:
                self.nodes.append(Node(self.simpy_env, 1, Position(size_x, size_y), NodeType.SENSOR))
            else:
                n = max(number_of_nodes-1,1)
                for x in range(1, number_of_nodes+1):
                    self.nodes.append(Node(self.simpy_env, x, Position(-size_x/2+(x-1)*size_x/n, -size_y/2+(x-1)*size_y/n), NodeType.SENSOR))

        elif positioning == "matrix":
            if n_x is None or n_y is None:
                print("Specify number of nodes in each direction.")

            uid = 1
            for y in range(0, n_y):
                for x in range(0, n_x):
                    self.nodes.append(Node(self.simpy_env, uid, Position(-size_x/2+x*size_x/(n_x-1), -size_y/2+y*size_y/(n_y-1)), NodeType.SENSOR))
                    uid += 1

        elif positioning == "matrix-random":
            rnd = kwargs.get("size_random")
            uid = 1
            for y in range(0, n_y):
                for x in range(0, n_x):
                    self.nodes.append(Node(self.simpy_env, uid, Position(-size_x / 2 + x * size_x / (n_x - 1) + np.random.uniform(-rnd/2, rnd),
                                                                         -size_y / 2 + y * size_y / (n_y - 1) + np.random.uniform(-rnd/2, rnd)),
                                           NodeType.SENSOR))
                    uid += 1

        self.link_table = LinkTable(self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.nodes, self.link_table)

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

    def pdr(self):
        payloads_sent = 0
        payloads_received = 0
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                payloads_sent += len(node.own_payloads_sent)
                payloads_received += len(node.own_payloads_arrived_at_gateway)

        return payloads_received/payloads_sent


