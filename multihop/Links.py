from .config import settings
from .utils import *
import numpy as np
import matplotlib

import networkx as nx


class LinkTable:
    def __init__(self, nodes):
        self.network = nx.Graph()

        self.link_table = {}
        for node1 in nodes:
            _link_table = {}
            for node2 in nodes:
                if node1.uid is not node2.uid:
                    if node1.uid < node2.uid:
                        _link_table[node2.uid] = Link(node1, node2)
                        self.network.add_node(node1.uid, pos=(node1.position.x, node1.position.y))
                        self.network.add_node(node2.uid, pos=(node2.position.x, node2.position.y))
                        if _link_table[node2.uid].in_range():
                            w = 0.5 + _link_table[node2.uid].rss() - settings.LORA_SENSITIVITY
                            self.network.add_edge(node1.uid, node2.uid, weight=w)
            self.link_table[node1.uid] = _link_table

    def get(self, node1, node2):
        return self.get_from_uid(node1.uid, node2.uid)

    def get_from_uid(self, node1_uid, node2_uid):
        # Sort to enable reciprocity: always the smallest first
        if node1_uid > node2_uid:
            x = node1_uid
            node1_uid = node2_uid
            node2_uid = x

        return self.link_table.get(node1_uid).get(node2_uid)

    def get_all_pairs_shortest_path(self):
        return list(nx.all_pairs_shortest_path(self.network))

    def plot(self, nw=None, w=16):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if nw is None:
            nw = self.network

        mpl.rc('image', cmap='Greys')
        fig, ax = plt.subplots()

        pos = nx.get_node_attributes(nw, 'pos')
        edges, weights = zip(*nx.get_edge_attributes(nw, 'weight').items())

        weights = tuple(item / w * 4 for item in weights)

        nx.draw(nw, pos, with_labels=True, width=weights)

        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.axis('equal')
        plt.show(block=False)

    def plot_usage(self):
        network = nx.Graph()
        _max = 0
        for i, links in self.link_table.items():
            for j, link in links.items():
                network.add_node(link.node1.uid, pos=(link.node1.position.x, link.node1.position.y))
                network.add_node(link.node2.uid, pos=(link.node2.position.x, link.node2.position.y))
                u = self.get_from_uid(link.node1.uid, link.node2.uid).used
                if u > _max:
                    _max = u
                network.add_edge(link.node1.uid, link.node2.uid, weight=u)

        self.plot(network, _max)


class Link:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self._shadowing = np.random.normal(settings.SHADOWING_MU, settings.SHADOWING_SIGMA)
        self.used = 0

        self._rss = 0
        self._snr = 0
        self._distance = 0
        self._in_range = False

    def rss(self):
        self._rss = settings.LORA_TRANSMIT_POWER - self.path_loss()
        return self._rss

    def distance(self):
        d = np.sqrt(np.abs(self.node1.position.x - self.node2.position.x) ** 2 + np.abs(
            self.node1.position.y - self.node2.position.y) ** 2)
        if d == 0:  # For when two nodes practically are at the same position
            d = 0.00001
        self._distance = d
        return d

    def path_loss(self):
        shadowing = 0
        if settings.SHADOWING_ENABLED:
            shadowing = self._shadowing
        return 74.85 + 2.75 * 10 * np.log10(self.distance()) + shadowing  # shadowing per link

    def snr(self):
        self._snr = self.rss() + 116.86714407 # thermal noise for 25°C 500kHz BW
        return self._snr

    def in_range(self):
        self._in_range = self.rss() > settings.LORA_SENSITIVITY
        return self._in_range

    def lqi(self):
        return settings.SNR_MAX - self.snr()

    def use(self):
        self.used += 1
