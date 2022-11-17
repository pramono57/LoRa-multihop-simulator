from config import settings
import utils
import numpy as np


class LinkTable:
    def __init__(self, nodes):
        self.link_table = {}
        for node1 in nodes:
            _link_table = {}
            for node2 in nodes:
                if node1.uid is not node2.uid:
                    _link_table[node2.uid] = Link(node1, node2)

            self.link_table[node1.uid] = _link_table

    def get(self, node1, node2):
        return self.get_from_uid(node1.uid,node2.uid)

    def get_from_uid(self, node1_uid, node2_uid):
        # Sort to enable reciprocity: always the smallest first
        if node1_uid > node2_uid:
            x = node1_uid
            node1_uid = node2_uid
            node2_uid = x

        return self.link_table.get(node1_uid).get(node2_uid)


class Link:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.shadowing = np.random.normal(settings.SHADOWING_MU, settings.SHADOWING_SIGMA)

    def rss(self):
        return settings.tp - self.path_loss()

    def distance(self):
        return np.sqrt(np.abs(self.node1.position.x - self.node2.position.x)**2 + np.abs(self.node1.position.y - self.node2.position.y)**2)

    def path_loss(self):
        return 74.85 + 2.75 * 10 * np.log10(self.distance()) #+ self.shadowing  # shadowing per link

    def snr(self):
        return self.rss() + 116.86714407  # thermal noise for 25°C 500kHz BW

    def in_range(self):
        return self.rss() > settings.sensitivity

    def lqi(self):
        return settings.SNR_MAX - self.snr()
