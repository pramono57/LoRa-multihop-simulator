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
import networkx as nx
import pickle
from collections.abc import Iterable

class Network:
    def __init__(self, **kwargs):
        self.nodes = []
        self.simpy_env = simpy.Environment()
        self.link_table = None

        # Copy network
        nw = kwargs.get("network", None)
        if nw is not None:
            for node in nw.nodes:
                self.nodes.append(Node(self.simpy_env,
                                       node.uid,
                                       Position(node.position.x, node.position.y),
                                       node.type))
        else:
            n_x = None
            n_y = None
            n = None

            positioning = kwargs.get('shape', None)
            size_x = kwargs.get('size_x', None)
            size_y = kwargs.get('size_y', None)

            size = kwargs.get('size', None)
            if size is not None:
                size_x = size
                size_y = size

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
                        self.nodes.append(Node(self.simpy_env, x,
                                               Position(-size_x/2+(x-1)*size_x/n + np.random.uniform(-rnd/2, rnd),
                                                        -size_y/2+(x-1)*size_y/n + np.random.uniform(-rnd/2, rnd)),
                                               NodeType.SENSOR))

            elif positioning == "matrix":
                uid = 1
                for y in range(0, n_y):
                    for x in range(0, n_x):
                        self.nodes.append(Node(self.simpy_env, uid,
                                               Position(-size_x / 2 + x * size_x / (n_x - 1) + np.random.uniform(-rnd/2, rnd),
                                                        -size_y / 2 + y * size_y / (n_y - 1) + np.random.uniform(-rnd/2, rnd)),
                                               NodeType.SENSOR))
                        uid += 1

            elif positioning == "circles-equal" or positioning == "circles":
                uid = 1

                center_x = 0
                center_y = 0

                n_circles = math.ceil(n_x/2+1)
                n_per_circle = 0
                if positioning == "circles-equal":
                    n_per_circle = n_y
                    n_circles = n_x

                total_area = 0
                for x in range(1, n_circles+1):
                    a = size_x/n_x*x
                    b = size_x/n_x*x

                    total_area += math.pi * a * b

                carry = 0
                for x in range(1, n_circles+1):
                    a = size_x/n_x*x
                    b = size_x/n_x*x

                    area = math.pi * a * b
                    if positioning == "circles":
                        n_per_circle = round(n_x * n_y / total_area * area)
                        if n_per_circle < 4:
                            carry = 4 - n_per_circle
                            n_per_circle = 4
                        elif n_per_circle > 4+carry:
                            n_per_circle -= carry
                            carry = 0

                    start_angle = 0
                    if positioning == "circles-equal":
                        start_angle = 360/n_per_circle/2*x
                    elif positioning == "circles":
                        start_angle = 360 / n_per_circle
                        # start_angle = 360/n_per_circle/(x % 2 + 1)

                    for c in range(0, n_per_circle):
                        angle = (start_angle + 360 / n_per_circle * c) % 360
                        _x = a * b / math.sqrt(b**2 + a**2 * (math.tan(math.radians(angle)))**2)
                        if 270 >= angle > 90:
                            _x = -_x
                        _y = _x * math.tan(math.radians(angle))
                        self.nodes.append(Node(self.simpy_env, uid,
                                               Position(_x + np.random.uniform(-rnd/2, rnd),
                                                        _y + np.random.uniform(-rnd/2, rnd)),
                                               NodeType.SENSOR))
                        if uid == 36:
                            print("trouble")
                        uid += 1


            elif positioning == "funnel":
                levels = kwargs.get('levels', None)
                d = math.sqrt(size_x**2 + size_y**2)/len(levels)
                i = 1
                base_angle = math.degrees(math.atan(size_y/size_x))
                for level, n_level in enumerate(levels):

                    circle_x = 0
                    circle_y = 0
                    circle_r = d * (level + 1)

                    # circle_x = d * level * math.cos(math.radians(base_angle))
                    # circle_y = d * level * math.sin(math.radians(base_angle))
                    # circle_r = d

                    angle_viewport = min(180, 90 / (2*max(1, level))+n_level*5)
                    if n_level > 1:
                        angle_start = -angle_viewport/2
                        angle_step = angle_viewport / (n_level - 1)
                    else:
                        angle_start = 0
                        angle_step = 0
                    for j in range(0, n_level):
                        # calculating coordinates
                        x = circle_r * math.cos(math.radians(base_angle + angle_start + j*angle_step)) + circle_x
                        y = circle_r * math.sin(math.radians(base_angle + angle_start + j*angle_step)) + circle_y

                        self.nodes.append(Node(self.simpy_env, i,
                                               Position(x + np.random.uniform(-rnd/2, rnd),
                                                        y + np.random.uniform(-rnd/2, rnd)),
                                               NodeType.SENSOR))

                        i += 1

            # recalc = self.evaluate_distances()
            # while recalc:
            #     recalc = self.evaluate_distances()

        self.link_table = LinkTable(self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.nodes, self.link_table)

    def update(self):
        self.link_table = LinkTable(self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.nodes, self.link_table)
    def add_sensor_node(self, uid, x, y):
        self.nodes.append(Node(self.simpy_env, uid,
                               Position(x, y),
                               NodeType.SENSOR))
        self.update()

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

    def rerun(self, time):
        self.simpy_env = simpy.Environment()
        for node in self.nodes:
            node.env = self.simpy_env
        self.run(time)

    def run(self, time):
        # First get max hop count in total network to establish how long the simulation should be extended
        paths = self.link_table.get_all_pairs_shortest_path()
        max_hops = 0
        for path in paths:
            hops = len(max(path[1].values(), key=len))
            if hops > max_hops:
                max_hops = hops

        # Extend simulation time
        simulation_time = time + (max_hops * (settings.TX_AGGREGATION_TIMER_NOMINAL
                                              + settings.TX_AGGREGATION_TIMER_STEP_UP
                                              * settings.TX_AGGREGATION_TIMER_MAX_TIMES_STEP_UP
                                              + settings.TX_AGGREGATION_TIMER_RANDOM[1]
                                              + settings.TX_COLLISION_TIMER_NOMINAL
                                              + settings.TX_COLLISION_TIMER_RANDOM[1]))

        # Start all nodes
        for node in self.nodes:
            node.sense_until = time
            self.simpy_env.process(node.run())

        self.simpy_env.run(until=simulation_time)

    def copy(self):
        return Network(network=self)

    def plot_network(self):
        self.link_table.plot()

    def plot_network_usage(self):
        self.link_table.plot_usage()

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
                    route = node.route.find_best()
                    hops = 0
                    if route is not None:
                        hops = route["hops"]
                    else: # If no route is yet found by multihop protocol, find location in networkx
                        hops = len(nx.shortest_path(self.link_table.network, source=0, target=node.uid))

                    ret = method(node)
                    if isinstance(ret, Iterable):
                        if data.get(hops, None) is None:
                            data[hops] = ret
                        else:
                            data[hops].extend(ret)
                    else:
                        if data.get(hops, None) is None:
                            data[hops] = [ret]
                        else:
                            data[hops].append(ret)

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

    def plot_aggregation_timer_values(self, v):
        import matplotlib.pyplot as plt
        data = {}
        for node in self.nodes:
            uid = node.uid

            hops = 0
            route = node.route.find_best()
            if route is not None:
                hops = route["hops"]
            else:  # If no route is yet found by multihop protocol, find location in networkx
                hops = len(nx.shortest_path(self.link_table.network, source=0, target=node.uid))

            data[uid] = {
                "hops": hops,
                "aggregation_timer_times": node.aggregation_timer_times,
                "aggregation_timer_values": node.aggregation_timer_values,
            }

        with open(f"results/aggregation_timer_{v}.pickle", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.figure()
        for uid, node in data.items():
            hops = node['hops']
            plt.plot(node["aggregation_timer_times"], node["aggregation_timer_values"], label=f"uid {uid}, hops {hops}")
        plt.show()

