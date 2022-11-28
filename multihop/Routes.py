from .config import settings
from tabulate import tabulate


class Route:
    def __init__(self):
        self.neighbour_list = []

    def update(self, uid, snr, cumulative_lqi, hops):
        # TODO temp fix
        neighbour = self.find_node(uid)
        if neighbour is None:
            if len(self.neighbour_list) >= settings.MAX_ROUTE_SIZE:
                self.neighbour_list.remove(self.find_worst())

            self.neighbour_list.append({'uid': uid,
                                        'snr': snr,
                                        'cumulative_lqi': cumulative_lqi,
                                        'hops': hops,
                                        'best': False})
        else:
            neighbour["uid"] = uid
            neighbour["snr"] = snr
            neighbour["cumulative_lqi"] = cumulative_lqi
            neighbour["hops"] = hops
        self.find_route()

    def find_node(self, _uid):
        for neighbour in self.neighbour_list:
            if _uid == neighbour["uid"]:
                return neighbour
        return None

    def find_worst(self):
        worst_i = 0
        for i, neighbour in enumerate(self.neighbour_list):
            if neighbour["cumulative_lqi"] > self.neighbour_list[worst_i]["cumulative_lqi"]:
                # cumulative LQI of this neighbour is worse than the previous one
                # -> save index of this neighbour
                worst_i = i
            elif neighbour["cumulative_lqi"] == self.neighbour_list[worst_i]["cumulative_lqi"]:
                # See if the LQI is equal -> worst route is the highest number of hops
                if neighbour["hops"] > self.neighbour_list[worst_i]["hops"]:
                    worst_i = i
                elif neighbour["hops"] == self.neighbour_list[worst_i]["hops"]:
                    # See if the nr of hops is equal -> worst route is the lowest snr to neighbour
                    if neighbour["snr"] < self.neighbour_list[worst_i]["snr"]:
                        worst_i = i

        return self.neighbour_list[worst_i]

    def find_best(self):
        if len(self.neighbour_list) > 0:
            best_i = 0
            for i, neighbour in enumerate(self.neighbour_list):
                neighbour["best"] = False
                if neighbour["cumulative_lqi"] < self.neighbour_list[best_i]["cumulative_lqi"]:
                    # cumulative LQI of this neighbour is better than the previous one
                    # -> save index of this neighbour
                    best_i = i
                elif neighbour["cumulative_lqi"] == self.neighbour_list[best_i]["cumulative_lqi"]:
                    # See if the LQI is equal -> best route is the lowest number of hops
                    if neighbour["hops"] < self.neighbour_list[best_i]["hops"]:
                        best_i = i
                    elif neighbour["hops"] == self.neighbour_list[best_i]["hops"]:
                        # See if the nr of hops is equal -> best route is the lowest snr to neighbour
                        if neighbour["snr"] > self.neighbour_list[best_i]["snr"]:
                            best_i = i

            self.neighbour_list[best_i]["best"] = True
            return self.neighbour_list[best_i]
        else:
            return None

    def find_route(self):
        return self.find_best()

    def __str__(self):
        return tabulate(self.neighbour_list, headers="keys")

