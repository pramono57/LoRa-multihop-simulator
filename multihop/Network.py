from .utils import *
from .config import settings
from .Timers import TxTimer, TimerType
from .Packets import Message, MessageType
from .Links import LinkTable

import numpy as np
import simpy
from aenum import Enum, MultiValue, auto, IntEnum
import collections
from tabulate import tabulate
import simpy



class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def random(cls, size):
        return Position(*np.random.uniform(-size/2, size/2, size=2))


class GatewayState(IntEnum):
    STATE_INIT = auto()
    STATE_ROUTE_DISCOVERY = auto()
    STATE_RX = auto()
    STATE_PROC = auto()


class NodeType(IntEnum):
    GATEWAY = auto()
    SENSOR = auto()

class Route:
    def __init__(self):
        self.neighbour_list = []

    def update(self, uid, snr, cumulative_lqi, hops):
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


class NodeState(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    STATE_INIT = 0, "INIT"
    STATE_CAD = 2, "CAD"
    STATE_RX = 3, "RX"
    STATE_TX = 6, "TX"
    STATE_PREAMBLE_TX = 5, "P_TX"
    STATE_SLEEP = 1, "ZZZ"
    STATE_SENSING = 4, "SNS"


def power_of_state(s: NodeState):
    if s is NodeState.STATE_INIT: return 0
    if s is NodeState.STATE_CAD: return settings.POWER_CAD_CYCLE_mW
    if s is NodeState.STATE_RX: return settings.POWER_RX_mW
    if s is NodeState.STATE_TX: return settings.POWER_TX_mW
    if s is NodeState.STATE_PREAMBLE_TX: return settings.POWER_TX_mW
    if s is NodeState.STATE_SLEEP: return settings.POWER_SLEEP_mW
    if s is NodeState.STATE_SENSING:
        return settings.POWER_SENSE_mW
    else:
        ValueError(f"Sensorstate {s} is unknown")


class Node:
    def __init__(self, env: simpy.Environment, _id, _position, _type):
        self.type = _type

        # Statistics
        self.collisions = []
        self.messages_sent = []
        self.messages_for_me = []
        self.own_payloads_sent = []
        self.own_payloads_arrived_at_gateway = []
        self.forwarded_payloads = []
        self.message_counter_own_data_and_forwarded_data = 0
        self.message_counter_only_forwarded_data = 0
        self.message_counter_only_own_data = 0
        
        # Routing and network
        self.link_table = None
        self.route = Route()
        self.nodes = []  # list containing the other nodes in the network

        # Buffers
        self.data_buffer = []
        self.forwarded_mgs_buffer = []
        self.route_discovery_forward_buffer = None
        self.messages_seen = collections.deque(maxlen=settings.MAX_SEEN_PACKETS)

        # State vars
        self.done_tx = 0
        self.states_time = []
        self.states = []
        self.message_in_tx = None

        # Properties
        self.uid = _id
        self.state = None
        self.env = env
        self.energy_mJ = 0
        self.time_to_sense = None

        # Timers
        self.tx_collision_timer = TxTimer(env, TimerType.COLLISION)
        if self.type == NodeType.GATEWAY:
            self.tx_route_discovery_timer = TxTimer(env, TimerType.ROUTE_DISCOVERY)
        else:
            self.tx_route_discovery_timer = None
        self.tx_aggregation_timer = TxTimer(env, TimerType.AGGREGATION)
        self.sense_timer = TxTimer(env, TimerType.SENSE)

        self.position = _position

        # if self.type == NodeType.GATEWAY:
        #     self.position = Position(0, 0)
        # else:
        #     self.position = Position.random(size=100)

        # Payload
        self.application_counter = 0

    def add_meta(self, nodes, link_table):
        self.nodes = nodes
        self.link_table = link_table

    def state_change(self, state_to):
        if state_to is self.state and state_to is not NodeState.STATE_SLEEP:
            print("mmm not possible, only sleepy can do this")
        # if self.state is None:
        #     print(f"{self.uid}\tState change: None->{state_to.fullname}")
        # else:
        #     print(f"{self.uid}\tState change: {self.state.fullname}->{state_to.fullname}")
        if state_to is not self.state:
            self.state = state_to
            self.states.append(state_to)
            self.states_time.append(self.env.now)

    def run(self):
        random_wait = np.random.uniform(0, settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        yield self.env.timeout(random_wait)
        print(f"{self.uid}\tStarting node {self.uid}")
        if self.type == NodeType.GATEWAY:
            self.tx_route_discovery_timer.start()
        else:
            self.sense_timer.start()

        self.state_change(NodeState.STATE_INIT)
        self.env.process(self.periodic_wakeup())

    def check_sensing(self):
        if self.sense_timer.is_expired():
            self.state_change(NodeState.STATE_SENSING)
            yield self.env.timeout(settings.MEASURE_DURATION_S)
            # schedule timer for transmit
            print(f"{self.uid}\tSensing")
            self.tx_aggregation_timer.start(restart=False)
            self.sense_timer.start()

            self.data_buffer.extend(self.application_counter.to_bytes(2, 'big'))
            self.application_counter = (self.application_counter + 1) % 65535

    def check_transmit(self):
        # Route discovery messages are sent separately
        if self.type == NodeType.GATEWAY:
            if self.tx_route_discovery_timer.is_expired():
                self.route_discovery_forward_buffer = \
                    Message(MessageType.TYPE_ROUTE_DISCOVERY, 0, 0, 0, 0, [0x55, 0x55, 0x55], self, [])
                yield self.env.process(self.tx(route_discovery=True))
                self.tx_route_discovery_timer.reset()
                #

        elif self.type == NodeType.SENSOR:
            # if route discovery message need to be forwarded (because of collision timer)
            if self.tx_collision_timer.is_expired():
                yield self.env.process(self.tx(route_discovery=True))
                self.tx_collision_timer.reset()
                #

            # elif to ensure beacon TX is not directly followed by a data TX
            # TODO ensure beacon TX does not throttle data TX
            elif self.tx_aggregation_timer.is_expired() or self.full_buffer():
                yield self.env.process(self.tx())
                self.tx_aggregation_timer.reset()

    def periodic_wakeup(self):
        while True:
            self.state_change(NodeState.STATE_SLEEP)
            cad_interval = random(settings.CAD_INTERVAL_RANDOM_S)
            yield self.env.timeout(cad_interval)

            cad_detected = yield self.env.process(self.cad())

            if cad_detected:
                packet_for_us = yield self.env.process(self.receiving())

            else:
                yield self.env.process(self.check_transmit())

            if self.type != NodeType.GATEWAY:
                yield self.env.process(self.check_sensing())

    def receiving(self):
        """
        After activity during CAD in periodic_cad, we will listen to incoming packets from other nodes

        :return:
        """
        packet_for_us = False
        active_node = None
        # print(f"{self.uid}\tChecking for RX packet")
        self.state_change(NodeState.STATE_RX)
        active_nodes = self.get_nodes_in_state(NodeState.STATE_PREAMBLE_TX)

        if len(active_nodes) > 0:
            if len(active_nodes) == 1:
                active_node = active_nodes[0]
            else:
                # if power higher than power_threshold for all tx nodes, this one will succeed
                power_threshold = 6  # dB
                powers = [(a, self.link_table.get(a, self).rss()) for a in active_nodes]
                powers.sort(key=lambda tup: tup[1], reverse=True)
                # only success for the highest power if > power_threshold
                if powers[0][1] >= powers[1][1] + power_threshold:
                    active_node = powers[0][0]
                else:
                    self.handle_collision(active_nodes)
                    active_node = None

            if active_node is not None:
                time_tx_done = active_node.done_tx
                yield self.env.timeout(abs(self.env.now - time_tx_done))
                rx_packet = active_node.message_in_tx

                packet_for_us = self.handle_rx_msg(rx_packet)

        return packet_for_us

    def tx(self, route_discovery: bool = False):
        # TODO do CAD before, and schedule TX for next time if channel is not free

        if route_discovery and self.route_discovery_forward_buffer is not None:
            self.message_in_tx = self.route_discovery_forward_buffer

        elif not route_discovery and ( len(self.data_buffer) > 0 or len(self.forwarded_mgs_buffer) > 0):
            # Init message for forwarding
            hops = 0
            lqi = 0
            if len(self.forwarded_mgs_buffer) > 0:
                hops = self.forwarded_mgs_buffer[0].header.hops
                lqi = self.forwarded_mgs_buffer[0].header.cumulative_lqi

            route = self.route.find_route()
            if route is None:
                route = 0
            else:
                route = route["uid"]

            self.message_in_tx = Message(MessageType.TYPE_ROUTED,
                                         hops,
                                         0,
                                         route,
                                         self.uid,
                                         self.data_buffer,
                                         self,
                                         self.forwarded_mgs_buffer)

            # Increase counters and adjust lqi if forward
            if len(self.message_in_tx.payload.forwarded_data) > 0:
                self.message_in_tx.hop(self)
                self.message_in_tx.header.cumulative_lqi += \
                    self.link_table.get_from_uid(self.uid, self.forwarded_mgs_buffer[0].payload.own_data.src).lqi()

        if self.message_in_tx is not None:
            self.state_change(NodeState.STATE_PREAMBLE_TX)

            packet_time = self.message_in_tx.time()
            self.done_tx = self.env.now + settings.PREAMBLE_DURATION_S + packet_time

            yield self.env.timeout(settings.PREAMBLE_DURATION_S)

            self.state_change(NodeState.STATE_TX)
            print(f"{self.uid}\t Sending packet {self.message_in_tx.header.uid} with size: {self.message_in_tx.size()} bytes")
            print(f"{self.uid}\tTx packet to {self.message_in_tx.header.address} {self.message_in_tx}")

            # Sent statistics
            self.messages_sent.append(self.message_in_tx)
            # Only for routed messages
            if self.message_in_tx.header.type == MessageType.TYPE_ROUTED:
                if self.message_in_tx.payload.own_data.len > 0:
                    self.own_payloads_sent.append(self.message_in_tx.payload.own_data)
                for pl in self.message_in_tx.payload.forwarded_data:
                    self.forwarded_payloads.append(pl)

                if self.message_in_tx.payload.own_data.len > 0 and len(self.message_in_tx.payload.forwarded_data) > 0:
                    self.message_counter_own_data_and_forwarded_data += 1
                elif self.message_in_tx.payload.own_data.len > 0:
                    self.message_counter_only_own_data += 1
                elif len(self.message_in_tx.payload.forwarded_data) > 0:
                    self.message_counter_only_forwarded_data += 1

            yield self.env.timeout(packet_time)
            self.done_tx = None
            self.message_in_tx = None
            self.forwarded_mgs_buffer = []
            self.data_buffer = []

            if route_discovery:
                self.route_discovery_forward_buffer = None

    def cad(self):
        """
        In the CAD state, the node listens for channel activity
        In the beginning it needs to wake-up and stabilise
        After that all messages sent in the CAD window, will be considered received (if power level is above sensitivity)

        Depending on CAD success the node enters RX state or sleep state
        """
        self.state_change(NodeState.STATE_CAD)
        yield self.env.timeout(settings.TIME_CAD_WAKE_S + settings.TIME_CAD_STABILIZE_S)

        # check which nodes are now in PREAMBLE_TX state
        nodes_sending_preamble = self.get_nodes_in_state(NodeState.STATE_PREAMBLE_TX)
        active_nodes = []
        for n in nodes_sending_preamble:
            if self.link_table.get(self, n).in_range():
                active_nodes.append(n)

        # check after CAD perform, if it was transmitting during the full window
        yield self.env.timeout(settings.TIME_CAD_PERFORM_S)

        cad_detected = False
        for n in active_nodes:
            if n.state is NodeState.STATE_PREAMBLE_TX:
                # OK considered TX and we need to listen
                cad_detected = True
                break
        yield self.env.timeout(settings.TIME_CAD_PROC_S)
        return cad_detected

    def full_buffer(self):
        return len(self.data_buffer) > settings.MAX_BUF_SIZE_BYTE

    def get_nodes_in_state(self, state):
        nodes = []
        for n in self.nodes:
            if n is not self and n.state is state:
                nodes.append(n)
        return nodes

    def handle_collision(self, active_nodes):
        for node in active_nodes:
            # All route discovery messages are usefull to us
            if node.message_in_tx.is_route_discovery():
                print(f"{self.uid}\t Route discovery collision detected")
                self.collisions.append(self.env.now)
            # Only routed messages, addressed to us are usefull 
            elif node.message_in_tx.is_routed() and node.message_in_tx.header.address == self.uid:
                print(f"{self.uid}\t Routed collision detected that was addressed to us")
                self.collisions.append(self.env.now)
        # Put collision tracking in Links?


    def handle_rx_msg(self, rx_packet):
        packet_for_us = False
        update_beacon = False
        # Check for routing in all received route discovery messages
        if rx_packet.is_route_discovery():
            print(f"{self.uid}\tRx packet from {rx_packet.payload.own_data.src} {rx_packet}")
            self.route.update(rx_packet.header.address,
                              self.link_table.get_from_uid(self.uid,
                                                           rx_packet.header.address).snr(),
                              rx_packet.header.cumulative_lqi + self.link_table.get_from_uid(self.uid, rx_packet.header.address).lqi(),
                              rx_packet.header.hops)
            print(f"{self.uid}\r\n{self.route}")

        if rx_packet.header.uid in self.messages_seen:
            print(
                f"{self.uid}\tPacket already processed {rx_packet.header.uid}")
            return False
        else:
            if rx_packet.is_route_discovery():
                packet_for_us = True
                self.route_discovery_forward_buffer = rx_packet.copy()
                # Update new packet that needs to be forwarded
                self.route_discovery_forward_buffer.header.address = self.uid
                self.route_discovery_forward_buffer.hop(self)
                self.route_discovery_forward_buffer.header.cumulative_lqi += \
                    self.link_table.get_from_uid(self.uid, rx_packet.header.address).lqi()
                self.tx_collision_timer.start(restart=True)

            elif rx_packet.is_routed() and rx_packet.header.address == self.uid:
                packet_for_us = True
                print(f"{self.uid}\tRx packet from {rx_packet.payload.own_data.src} {rx_packet}")
                # update tx timer
                self.messages_for_me.append(rx_packet)
                if self.type == NodeType.SENSOR:
                    self.tx_aggregation_timer.start(restart=False)  
                    self.tx_aggregation_timer.step_up() # Is only applied for for next start
                    self.forwarded_mgs_buffer.append(rx_packet)
                elif self.type == NodeType.GATEWAY:
                    rx_packet.arrived_at_gateway()

            self.messages_seen.append(rx_packet.header.uid)

        return packet_for_us

    def arrived_at_gateway(self, payload): 
        # Update statistics
        self.own_payloads_arrived_at_gateway.append(payload)

    def pdr(self):
        return len(self.own_payloads_arrived_at_gateway)/len(self.own_payloads_sent)

    def plot_states(self, axis=None, plot_labels=True):
        import matplotlib.pyplot as plt

        state_time = []
        states = []

        for i, (t, s) in enumerate(zip(self.states_time, self.states)):
            states.append(int(s.value))
            state_time.append(t)

            if i < len(self.states_time) - 1:
                # state actually till next transition time
                states.append(int(s.value))
                state_time.append(self.states_time[i + 1])

        if axis is None:
            axis = plt.subplot()

        if self.type == NodeType.GATEWAY:
            axis.plot(state_time, states, color="red")
        else:
            axis.plot(state_time, states)
        axis.scatter(self.collisions, [0]*len(self.collisions), edgecolor="green")
        axis.grid()

        # if plot_labels:
        #
        #     states = sorted(self.states, key=lambda tup: tup.value)
        #     y = [s.value for s in states]
        #     labels = [f"{s.fullname}" for s in states]
        #
        #     axis.set_yticks(y, labels)
        if axis is None:
            plt.show()

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
            if n_x == None or n_y == None:
                print("Specify number of nodes in each direction.")

            uid = 1
            for y in range(0, n_y):
                for x in range(0, n_x):
                    self.nodes.append(Node(self.simpy_env, uid, Position(-size_x/2+x*size_x/(n_x-1), -size_y/2+y*size_y/(n_y-1)), NodeType.SENSOR))
                    uid += 1

        self.link_table = LinkTable(self.nodes)
        for node in self.nodes:
            if type(node) is Node:
                node.add_meta(self.nodes, self.link_table)

    def run(self, time):
        for node in self.nodes:
            self.simpy_env.process(node.run())
        self.simpy_env.run(until=time)

    def plot_network(self):
        self.link_table.plot()

    def plot_states(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(self.nodes), sharex=True, sharey=True)
        for i, node in enumerate(self.nodes):
            node.plot_states(ax[i])

        ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6], ["INIT", "ZZZ", "CAD", "RX", "SNS", "P_TX", "TX"])
        plt.show()

    def pdr(self):
        payloads_sent = 0
        payloads_received = 0
        for node in self.nodes:
            if node.type == NodeType.SENSOR:
                payloads_sent += len(node.own_payloads_sent)
                payloads_received += len(node.own_payloads_arrived_at_gateway)

        return payloads_received/payloads_sent


