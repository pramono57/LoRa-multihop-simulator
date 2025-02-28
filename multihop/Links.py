from .utils import *
import numpy as np
import matplotlib

import networkx as nx


class LinkTable:
    def __init__(self, settings, nodes):
        self.network = nx.Graph()
        self.settings = settings

        self.link_table = {}
        for node1 in nodes:
            _link_table = {}
            for node2 in nodes:
                if node1.uid is not node2.uid:
                    if node1.uid < node2.uid:
                        _link_table[node2.uid] = Link(self.settings, node1, node2)
                        self.network.add_node(node1.uid, name="{:02d}".format(node1.uid), pos=(node1.position.x+50, node1.position.y+100))
                        self.network.add_node(node2.uid, name="{:02d}".format(node2.uid), pos=(node2.position.x+50, node2.position.y+100))
                        if _link_table[node2.uid].in_range():
                            w = 1 + _link_table[node2.uid].rss() - self.settings.LORA_SENSITIVITY[self.settings.LORA_SF]
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

        pos = nx.get_node_attributes(nw, 'pos')
        edges, weights = zip(*nx.get_edge_attributes(nw, 'weight').items())
        weights = tuple(item / w * 4 for item in weights)

        # from network2tikz import plot as network_plot_tikz
        # style = {}
        # style['vertex_label'] = nx.get_node_attributes(self.network, 'name')
        # style['edge_curved'] = 0.1
        # style["canvas"] = (10, 10)
        # style['layout'] = pos
        # style['edge_width'] = {e:0.3 * f for e,f in nx.get_edge_attributes(self.network,'weight').items()}
        # style['keep_aspect_ratio'] = True
        # style['vertex_size'] = .8
        # style['vertex_color'] = "gray!10"
        #
        # network_plot_tikz(self.network, './results/network.tex', **style)

        mpl.rc('image', cmap='Greys')
        fig, ax = plt.subplots()


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
    def __init__(self, settings, node1, node2):
        self.settings = settings

        self.node1 = node1
        self.node2 = node2
        self._shadowing = np.random.normal(self.settings.SHADOWING_MU, self.settings.SHADOWING_SIGMA)
        self.used = 0

        self._rss = 0
        self._snr = 0
        self._valid_snr = False
        self._distance = 0
        self._in_range = False


    def rss(self):
        self._rss = self.settings.LORA_TRANSMIT_POWER - self.path_loss()
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
        if self.settings.SHADOWING_ENABLED:
            shadowing = self._shadowing

        if self.settings.ENVIRONMENT.lower() == "urban":
            return 74.85 + 2.75 * 10 * np.log10(self.distance()) + shadowing  # shadowing per link
        elif self.settings.ENVIRONMENT.lower() == "coast":
            return 42.96 + 3.62 * 10 * np.log10(self.distance()) + shadowing  # shadowing per link
        elif self.settings.ENVIRONMENT.lower() == "forest":
            return 95.52 + 2.03 * 10 * np.log10(self.distance()) + shadowing  # shadowing per link

    def snr(self):
        #self._snr = self.rss() - -174 + 10 * np.log10(125e3)) # SNR = RSS - NOISE FLOOR
        self._snr = self.rss() + 174 - 10 * np.log10(self.settings.LORA_BANDWIDTH * 1e3) # thermal noise for 25°C 500kHz BW
        return self._snr

    def in_range(self):
        self._in_range = self.rss() > self.settings.LORA_SENSITIVITY[self.settings.LORA_SF]+4
        return self._in_range

    def valid_snr(self):
        self._valid_snr = self.snr() > self.settings.SNR_MIN_REQUIRED[self.settings.LORA_SF]
        return self._valid_snr

    def lqi(self):
        snr_limit = self.settings.SNR_MIN_REQUIRED[self.settings.LORA_SF] + self.settings.SNR_MARGIN
        return max(0, snr_limit - self.snr())

    def use(self):
        self.used += 1
