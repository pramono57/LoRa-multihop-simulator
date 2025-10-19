from .utils import *
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
import matplotlib as mpl
import copy
import sys
import tikzplotlib
from datetime import datetime
import dill as dill
import json

class Network:
    def __init__(self, **kwargs):
        print('\nNetwork -> init')
        self.nodes = []
        self.simpy_env = simpy.Environment()
        self.link_table = None

        self.start_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        s = kwargs.get("settings", None)
        if s is not None:
            self.settings = s
        else:
            from .config import settings as settings_from_file
            self.settings = settings_from_file

        self.settings["network"] = kwargs

    # Copy network
        nw = kwargs.get("network", None)
        map = kwargs.get("map", None)
        if nw is not None:
            for node in nw.nodes:
                self.nodes.append(Node(self.simpy_env,
                                       self.settings,
                                       node.uid,
                                       Position(node.position.x, node.position.y),
                                       node.type))
        elif map is not None:
            for uid, node in map.items():
                self.nodes.append(Node(self.simpy_env,
                                       self.settings,
                                       uid,
                                       Position(node["x"], node["y"]),
                                       node["type"]))
        else:
            n_x = None
            n_y = None
            n = None

            positioning = None
            if "NETWORK_SHAPE" in self.settings.keys():
                positioning = self.settings["NETWORK_SHAPE"]
            else:
                positioning = kwargs.get('shape', None)

            shape_file = None
            if "NETWORK_SHAPE_FILE" in self.settings.keys():
                shape_file = self.settings["NETWORK_SHAPE_FILE"]
            else:
                shape_file = kwargs.get('shape_file', None)

            size_x = None
            if "NETWORK_SIZE_X" in self.settings.keys():
                size_x = self.settings["NETWORK_SIZE_X"]
            else:
                size_x = kwargs.get('size_x', None)

            size_y = None
            if "NETWORK_SIZE_Y" in self.settings.keys():
                size_y = self.settings["NETWORK_SIZE_Y"]
            else:
                size_y = kwargs.get('size_y', None)

            size = kwargs.get('size', None)
            if size is None and "NETWORK_SIZE" in self.settings.keys():
                size = self.settings["NETWORK_SIZE"]

            if size is not None:
                size_x = size
                size_y = size

            density = None
            if "NETWORK_DENSITY" in self.settings.keys():
                density = self.settings["NETWORK_DENSITY"]
            else:
                density = kwargs.get('density', None)

            if density is not None:
                n_x = round(math.sqrt(density) / 1000 * size_x)
                n_y = round(math.sqrt(density) / 1000 * size_y)
                n = n_x * n_y
            else:
                n_x = kwargs.get('n_x', None)
                n_y = kwargs.get('n_y', None)
                if n_x is None or n_y is None:
                    n = kwargs.get('n', None)
                else:
                    n = n_x * n_y

            rnd = None
            if "NETWORK_SIZE_RANDOM" in self.settings.keys():
                rnd = self.settings["NETWORK_SIZE_RANDOM"]
            else:
                rnd = kwargs.get('size_random', None)
            if rnd is None:
                rnd = 0

            g_x = None
            if "NETWORK_SIZE_RANDOM" in self.settings.keys():
                g_x = self.settings["NETWORK_GATEWAY_X"]
            else:
                g_x = kwargs.get('g_x', None)
            if g_x is None:
                g_x = 0

            g_y = None
            if "NETWORK_SIZE_RANDOM" in self.settings.keys():
                g_y = self.settings["NETWORK_GATEWAY_Y"]
            else:
                g_y = kwargs.get('g_y', None)
            if g_y is None:
                g_y = 0

            fixed_route = kwargs.get('fixed_route', None)
            if fixed_route is not None:
                fixed_route = flatten_node_tree(fixed_route)
                node_uids = list(fixed_route.keys())
                n = len(node_uids)
            else:
                if positioning != "custom":
                    node_uids = range(0, n + 1)

            self.nodes.append(
                Node(self.simpy_env, self.settings, 0, Position(float(g_x), float(g_y)), NodeType.GATEWAY))

            if positioning == "random":
                for x in node_uids[1:]:
                    self.nodes.append(
                        Node(self.simpy_env, self.settings, x,
                             Position(*np.random.uniform(-size_x / 2, size_y / 2, size=2)),
                             NodeType.SENSOR, fixed_route=fixed_route))

            elif positioning == "line":
                if n == 1:
                    self.nodes.append(Node(self.simpy_env, self.settings, 1, Position(size_x, size_y), NodeType.SENSOR))
                else:
                    for x in node_uids[1:]:
                        self.nodes.append(Node(self.simpy_env, self.settings, x,
                                               Position(-size_x / 2 + (x - 1) * size_x / n + np.random.uniform(-rnd / 2,
                                                                                                               rnd),
                                                        -size_y / 2 + (x - 1) * size_y / n + np.random.uniform(-rnd / 2,
                                                                                                               rnd)),
                                               NodeType.SENSOR, fixed_route=fixed_route))

            elif positioning == "matrix":
                uid = 1
                for y in range(0, n_y):
                    for x in range(0, n_x):
                        self.nodes.append(Node(self.simpy_env, self.settings, node_uids[uid],
                                               Position(
                                                   -size_x / 2 + x * size_x / (n_x - 1) + np.random.uniform(-rnd / 2,
                                                                                                            rnd),
                                                   -size_y / 2 + y * size_y / (n_y - 1) + np.random.uniform(-rnd / 2,
                                                                                                            rnd)),
                                               NodeType.SENSOR, fixed_route=fixed_route))
                        uid += 1

            elif positioning == "circles-equal" or positioning == "circles":
                uid = 1

                center_x = 0
                center_y = 0

                n_circles = math.ceil(n_x / 2 + 1)
                n_per_circle = 0
                if positioning == "circles-equal":
                    n_per_circle = n_y
                    n_circles = n_x

                total_area = 0
                for x in range(1, n_circles + 1):
                    a = size_x / n_x * x
                    b = size_x / n_x * x

                    total_area += math.pi * a * b

                carry = 0
                for x in range(1, n_circles + 1):
                    a = size_x / n_x * x
                    b = size_x / n_x * x

                    area = math.pi * a * b
                    if positioning == "circles":
                        n_per_circle = round(n_x * n_y / total_area * area)
                        if n_per_circle < 4:
                            carry = 4 - n_per_circle
                            n_per_circle = 4
                        elif n_per_circle > 4 + carry:
                            n_per_circle -= carry
                            carry = 0

                    start_angle = 0
                    if positioning == "circles-equal":
                        start_angle = 360 / n_per_circle / 2 * x
                    elif positioning == "circles":
                        start_angle = 360 / n_per_circle
                        # start_angle = 360/n_per_circle/(x % 2 + 1)

                    for c in range(0, n_per_circle):
                        angle = (start_angle + 360 / n_per_circle * c) % 360
                        _x = a * b / math.sqrt(b ** 2 + a ** 2 * (math.tan(math.radians(angle))) ** 2)
                        if 270 >= angle > 90:
                            _x = -_x
                        _y = _x * math.tan(math.radians(angle))
                        self.nodes.append(Node(self.simpy_env, self.settings, node_uids[uid],
                                               Position(_x + np.random.uniform(-rnd / 2, rnd),
                                                        _y + np.random.uniform(-rnd / 2, rnd)),
                                               NodeType.SENSOR, fixed_route=fixed_route))
                        uid += 1


            elif positioning == "funnel":
                levels = kwargs.get('levels', None)
                d = math.sqrt(size_x ** 2 + size_y ** 2) / len(levels)
                i = 1
                base_angle = math.degrees(math.atan(size_y / size_x))
                for level, n_level in enumerate(levels):

                    circle_x = 0
                    circle_y = 0
                    circle_r = d * (level + 1)

                    # circle_x = d * level * math.cos(math.radians(base_angle))
                    # circle_y = d * level * math.sin(math.radians(base_angle))
                    # circle_r = d

                    angle_viewport = min(180, 90 / (2 * max(1, level)) + n_level * 5)
                    if n_level > 1:
                        angle_start = -angle_viewport / 2
                        angle_step = angle_viewport / (n_level - 1)
                    else:
                        angle_start = 0
                        angle_step = 0
                    for j in range(0, n_level):
                        # calculating coordinates
                        x = circle_r * math.cos(math.radians(base_angle + angle_start + j * angle_step)) + circle_x
                        y = circle_r * math.sin(math.radians(base_angle + angle_start + j * angle_step)) + circle_y

                        self.nodes.append(Node(self.simpy_env, self.settings, node_uids[i],
                                               Position(x + np.random.uniform(-rnd / 2, rnd),
                                                        y + np.random.uniform(-rnd / 2, rnd)),
                                               NodeType.SENSOR, fixed_route=fixed_route))

                        i += 1

            elif positioning == "custom":
                f = open(shape_file)
                nodes = json.load(f)

                for n in nodes:
                    if n["uid"] > 0:
                        self.nodes.append(Node(self.simpy_env, self.settings, n["uid"],
                                               Position(n["position"]["x"],
                                                        n["position"]["y"]),
                                               NodeType.SENSOR, fixed_route=fixed_route))

            recalc = self.evaluate_distances()
            while recalc:
                recalc = self.evaluate_distances()

        self.update()

    def set_settings(self, settings):
        self.settings = settings
        self.update()

    def get_node_map(self):
        print('\nNetwork -> get_node_map')
        map = {}
        for node in self.nodes:
            print(f'uid: {node.uid}, x: {node.position.x}, y: {node.position.y}, type: {node.type}')
            map[node.uid] = {"x": node.position.x, "y": node.position.y, "uid": node.uid, "type": node.type}
        return map

    def update(self):
        print('\nNetwork -> update')
        self.link_table = LinkTable(self.settings, self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.settings, self.nodes, self.link_table)

    def add_sensor_node(self, uid, x, y):
        print('\nNetwork -> add_sensor_node')
        self.nodes.append(Node(self.simpy_env, uid,
                               Position(x, y),
                               NodeType.SENSOR))
        self.update()

    def evaluate_distances(self):
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1.uid is not node2.uid and node1.uid < node2.uid:
                    d = np.sqrt(np.abs(node1.position.x - node2.position.x) ** 2 +
                                np.abs(node1.position.y - node2.position.y) ** 2)
                    if d < 1:
                        # Distance between two nodes is too small: move each node half of what is needed
                        t = -(1 - d) / 2
                        x1 = (1 - t / d) * node1.position.x + t / d * node2.position.x
                        y1 = (1 - t / d) * node1.position.y + t / d * node2.position.y
                        t = d + (1 - d) / 2
                        x2 = (1 - t / d) * node1.position.x + t / d * node2.position.x
                        y2 = (1 - t / d) * node1.position.y + t / d * node2.position.y

                        node1.position = Position(x1, y1)
                        node2.position = Position(x2, y2)

                        return True

                    print(f'Node: {node1.uid, node2.uid}, Distance: {d}')

    def rerun(self, time):
        self.simpy_env = simpy.Environment()
        for node in self.nodes:
            node.env = self.simpy_env
        self.run(time)

    def run(self):
        print('Network -> run')

        time = self.settings.SIMULATION_RUN_TIME

        # First get max hop count in total network to establish how long the simulation should be extended
        paths = self.link_table.get_all_pairs_shortest_path()
        max_hops = 0
        for path in paths:
            hops = len(max(path[1].values(), key=len))
            if hops > max_hops:
                max_hops = hops

            print(f'hops: {hops}, max_hops: {max_hops}')

        # Extend simulation time
        simulation_time = time + (max_hops * (self.settings.TX_AGGREGATION_TIMER_NOMINAL
                                              + self.settings.TX_AGGREGATION_TIMER_STEP_UP
                                              * self.settings.TX_AGGREGATION_TIMER_MAX_TIMES_STEP_UP
                                              + self.settings.TX_AGGREGATION_TIMER_RANDOM[1]
                                              + self.settings.TX_COLLISION_TIMER_NOMINAL
                                              + self.settings.TX_COLLISION_TIMER_RANDOM[1]))

        # Start all nodes
        for node in self.nodes:
            node.sense_until = time
            self.simpy_env.process(node.run())

        self.simpy_env.run(until=simulation_time)

    def extract_simpy(self):
        self.simpy_env = None
        for node in self.nodes:
            node.env = None
            node.sense_timer.env = None
            node.tx_aggregation_timer.env = None
            node.tx_collision_timer.env = None
            if node.tx_route_discovery_timer is not None:
                node.tx_route_discovery_timer.env = None

    def copy(self):
        return Network(network=self, settings=copy.deepcopy(self.settings))

    def plot_network(self):
        print('Network -> plot_network')
        self.link_table.plot()

    def plot_network_usage(self):
        self.link_table.plot_usage()

    def plot_states(self, ns=None):
        import matplotlib.pyplot as plt

        if ns is None:
            ns = len(self.nodes)

        fig, ax = plt.subplots(len(ns), sharex=True, sharey=True)
        i = 0
        for node in self.nodes:
            if node.uid in ns:
                node.plot_states(ax[i])
                i += 1

        ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6], ["INIT", "ZZZ", "CAD", "RX", "SNS", "P_TX", "TX"])
        plt.show(block=False)

    def statistic(self, stat0, stat, **kwargs):
        print('Network -> statistic')
        method = methodcaller(stat)

        data = {}
        for node in self.nodes:
            print(f'type: {node.type}')
            if node.type == NodeType.SENSOR:
                if node.route is not None:
                    hops = 0  # hops is legacy name and is used for stat0
                    if stat0 == "hops":
                        route = node.route.find_best()
                        if route is not None:
                            hops = route["hops"]
                        else:  # If no route is yet found by multi-hop protocol, find location in networkx
                            hops = len(nx.shortest_path(self.link_table.network, source=0, target=node.uid))
                    elif stat0 == "children":
                        hops = len(list(set(node.forwarded_from)))

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

        relative = kwargs.get("relative")
        if relative == "min":
            _min = sys.maxsize
            for hops, l in data.items():
                m = min(l)
                if m < _min:
                    _min = m
            for hops, l in data.items():
                data[hops] = [x / _min for x in l]
        elif relative == "max":
            _max = 0
            for hops, l in data.items():
                m = max(l)
                if m > _max:
                    _max = m
            for hops, l in data.items():
                data[hops] = [x / _max for x in l]

        data = dict(sorted(data.items()))

        return data

    def plot_hops_statistic(self, stat, **kwargs):
        import matplotlib.pyplot as plt

        data = self.statistic("children", stat, **kwargs)

        type = kwargs.get("type")

        fig, ax = plt.subplots()
        if type == "boxplot" or type is None:
            labels, pltdata = [*zip(*data.items())]  # 'transpose' items to parallel key, value lists
            plt.boxplot(pltdata)
            plt.xticks(range(1, len(labels) + 1), labels)
            ax.set_ylabel(stat)
        elif type == "cdf":
            mpl.use('TkAgg')
            fig, ax = plt.subplots(figsize=(8, 4))

            pdr = []
            for hops, d in data.items():
                pdr = pdr + d

            x_axis = [0] + np.sort(pdr).tolist()
            y_axis = [0] + (np.arange(len(pdr)) / (len(pdr) - 1)).tolist()
            plt.plot(x_axis, y_axis, label='overall')

            _filter = [[0], [1], [2], [3], [4], [5], [6], [7]]
            for f in _filter:
                pdr = []
                for mf in f:
                    pdr = pdr + data[mf]

                x_axis = [0] + np.sort(pdr).tolist()
                y_axis = [0] + (np.arange(len(pdr)) / (len(pdr) - 1)).tolist()
                plt.plot(x_axis, y_axis, label=''.join(str(x) for x in f))

            # for hop_count, x in data.items():
            #     x_axis = [0] + np.sort(x).tolist()
            #     y_axis = [0] + (np.arange(len(x)) / (len(x) - 1)).tolist()
            #     plt.plot(x_axis, y_axis, label=hop_count)

            #     n, bins, patches = ax.hist(np.sort(x), density=True, histtype='step', cumulative=True, label=hop_count)
            #     patches[0].set_xy(patches[0].get_xy()[:-1])
            # ax.set_xlabel(stat)
            # ax.legend()

        with open(f"results/{self.start_time}_{stat}.json", 'w') as fp:
            json.dump(data, fp)

        # tikzplotlib.save(f"results/{self.start_time}_{stat}.tex")
        plt.legend()
        plt.show(block=False)

    def pdr(self):
        payloads_sent = 0
        payloads_received = 0
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                payloads_sent += len(node.own_payloads_sent)
                payloads_received += len(node.own_payloads_arrived_at_gateway)

        pdr = payloads_received / payloads_sent
        print('\nNetwork -> pdr')
        print(f'Packet Delivery Ratio (PDR): {pdr}')
        return pdr

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

        with open(f"results/{self.start_time}aggregation_timer_values.json", 'w') as fp:
            json.dump(data, fp)
        tikzplotlib.save(f"results/{self.start_time}aggregation_timer_values.tex")

        plt.show()

    def hops_statistic_energy_per_state(self, normalized=True):
        data = {}
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                if node.route is not None:
                    route = node.route.find_best()
                    hops = 0
                    if route is not None:
                        hops = route["hops"]
                    else:  # If no route is yet found by multihop protocol, find location in networkx
                        hops = len(nx.shortest_path(self.link_table.network, source=0, target=node.uid))

                    ret = node.energy_per_state()
                    if normalized:
                        sum = 0
                        for state, value in ret.items():
                            sum += value

                        for state, value in ret.items():
                            ret[state] = value / sum

                    if hops not in data:
                        data[hops] = {}
                        for state in NodeState:
                            data[hops][state.name] = []
                    for state in NodeState:
                        data[hops][state.name].append(ret[state.name])
        return data

    def plot_hops_statistic_energy_per_state(self, normalized=True):
        data = self.hops_statistic_energy_per_state(normalized)

        import matplotlib.pyplot as plt
        plt.figure()
        figure, axis = plt.subplots(1, len(data))

        for hop, d in data.items():
            pie_values = []
            pie_labels = []
            for state, l in d.items():
                pie_values.append(np.average(l))
                pie_labels.append(state)
            axis[hop].pie(np.array(pie_values), labels=pie_labels)

        with open(f"results/{self.start_time}_hops_statistic_energy_per_state.json", 'w') as fp:
            json.dump(data, fp)
        tikzplotlib.save(f"results/{self.start_time}_hops_statistic_energy_per_state.tex")

        plt.show()

    def save_settings(self):
        with open(f"results/{self.start_time}_settings.json", 'w') as fp:
            json.dump(self.settings, fp)

    def save(self):
        self.extract_simpy()
        ofile = open(f"results/{self.start_time}_network.dill", "wb")
        dill.dump(self, ofile)
        ofile.close()

    @staticmethod
    def load(filename):
        ifile = open(filename, "rb")
        newdata = dill.load(ifile)
        ifile.close()

        return newdata
