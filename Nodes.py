import utils
from utils import random

import numpy as np
import simpy
from aenum import Enum, MultiValue, auto, IntEnum

from config import settings

from utils import random
from Timers import TxTimer, TimerType

import collections

from Packets import Message, MessageType
from copy import copy


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def random(cls, size):
        return Position(*np.random.uniform(0, size, size=2))


class GatewayState(IntEnum):
    STATE_INIT = auto()
    STATE_ROUTE_DISCOVERY = auto()
    STATE_RX = auto()
    STATE_PROC = auto()


class Gateway:
    def __init__(self, env: simpy.Environment, _id):
        self.uid = _id
        self.env = env
        self.route_discovery_message = Message(MessageType.TYPE_ROUTE_DISCOVERY, 0, 0, 0, 0, [0x55, 0x55, 0x55], [])
        self.message_in_tx = None
        self.state = SensorNodeState.STATE_INIT
        self.done_tx = None
        self.position = Position(0, 0)
        self.states_time = []
        self.states = []

    def state_change(self, state_to):
        if self.state is None:
            print(f"GW\tState change: None->{state_to.fullname}")
        else:
            print(f"GW\tState change: {self.state.fullname}->{state_to.fullname}")
        if state_to is not self.state:
            self.state = state_to
            self.states.append(state_to)
            self.states_time.append(self.env.now)

    def run(self):
        yield self.env.timeout(2 * settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        print(f"{self.uid}\tStarting GW {self.uid}")

        while True:
            #TODO include CAD before Tx

            self.state_change(SensorNodeState.STATE_PREAMBLE_TX)
            self.message_in_tx = self.route_discovery_message

            message_time = self.message_in_tx.time()
            self.done_tx = self.env.now + settings.PREAMBLE_DURATION_S + message_time

            yield self.env.timeout(settings.PREAMBLE_DURATION_S)

            self.state_change(SensorNodeState.STATE_TX)
            yield self.env.timeout(message_time)

            self.state_change(SensorNodeState.STATE_SLEEP)
            yield self.env.timeout(settings.GW_BEACON_INTERVAL_S)
            print("Sending Beacon from GW")

    def plot_states(self, axis=None):
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

        axis.plot(state_time, states, 'r')
        axis.grid()

        if axis is None:
            plt.show()


class SensorNodeState(Enum):
    _init_ = 'value fullname'
    _settings_ = MultiValue

    STATE_INIT = 0, "INIT"
    STATE_CAD = 2, "CAD"
    STATE_RX = 3, "RX"
    STATE_TX = 6, "TX"
    STATE_PREAMBLE_TX = 5, "P_TX"
    STATE_SLEEP = 1, "ZZZ"
    STATE_SENSING = 4, "SNS"


def power_of_state(s: SensorNodeState):
    if s is SensorNodeState.STATE_INIT: return 0
    if s is SensorNodeState.STATE_CAD: return settings.POWER_CAD_CYCLE_mW
    if s is SensorNodeState.STATE_RX: return settings.POWER_RX_mW
    if s is SensorNodeState.STATE_TX: return settings.POWER_TX_mW
    if s is SensorNodeState.STATE_PREAMBLE_TX: return settings.POWER_TX_mW
    if s is SensorNodeState.STATE_SLEEP: return settings.POWER_SLEEP_mW
    if s is SensorNodeState.STATE_SENSING:
        return settings.POWER_SENSE_mW
    else:
        ValueError(f"Sensorstate {s} is unknown")


class SensorNode:
    def __init__(self, env: simpy.Environment, _id):
        self.collisions = []
        self.data_buffer = []
        self.forwarded_mgs_buffer = []
        self.route_discovery_forward_buffer = None

        self.done_tx = 0
        self.states_time = []
        self.states = []
        self.message_in_tx = None

        self.uid = _id
        self.state = None
        self.env = env
        self.energy_mJ = 0
        self.time_to_sense = None

        self.shortest_path_dst = None
        self.shortest_path_num_hops = None
        self.shortest_path_SNR = None
        self.best_beacon = None
        self.beacon_seen_id = -1  # holds the highest beacon counter, seen (to ensure no duplicates)

        self.seen_packets = collections.deque(maxlen=settings.MAXMAX_SEEN_PACKETS)

        self.tx_collision_timer = TxTimer(env, TimerType.COLLISION)
        self.tx_aggregation_timer = TxTimer(env, TimerType.AGGREGATION)
        self.sense_timer = TxTimer(env, TimerType.SENSE)

        self.nodes = []  # list containing the other nodes in the network

        self.position = Position.random(size=100)
        self.best_route = 1  # needs to be populated through routing protocol

        self.application_counter = 0

    def add_nodes(self, nodes):
        self.nodes = nodes

    def state_change(self, state_to):
        if state_to is self.state and state_to is not SensorNodeState.STATE_SLEEP:
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
        self.sense_timer.start()

        self.state_change(SensorNodeState.STATE_INIT)
        self.env.process(self.periodic_wakeup())

    def check_sensing(self):
        if self.sense_timer.is_expired():
            self.state_change(SensorNodeState.STATE_SENSING)
            yield self.env.timeout(settings.MEASURE_DURATION_S)
            # schedule timer for transmit
            print(f"{self.uid}\tSensing")
            self.tx_aggregation_timer.start(restart=False)
            self.sense_timer.start()

            self.data_buffer.extend(self.application_counter.to_bytes(2, 'big')) 
            self.application_counter = (self.application_counter + 1) % 65535

    def check_transmit(self):
        # Route discovery messages are sent separately

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
            self.state_change(SensorNodeState.STATE_SLEEP)
            cad_interval = random(settings.CAD_INTERVAL_RANDOM_S)
            yield self.env.timeout(cad_interval)

            cad_detected = yield self.env.process(self.cad())

            if cad_detected:
                packet_for_us = yield self.env.process(self.receiving())


            else:
                yield self.env.process(self.check_transmit())

            yield self.env.process(self.check_sensing())

    def receiving(self):
        """
        After activity during CAD in periodic_cad, we will listen to incoming packets from other nodes

        :return:
        """
        packet_for_us = False
        active_node = None
        print(f"{self.uid}\tChecking for RX packet")
        self.state_change(SensorNodeState.STATE_RX)
        active_nodes = self.get_nodes_in_state(SensorNodeState.STATE_PREAMBLE_TX)

        if len(active_nodes) > 0:
            if len(active_nodes) == 1:
                active_node = active_nodes[0]
            else:
                # if power higher than power_threshold for all tx nodes, this one will succeed
                power_threshold = 6  # dB
                powers = [(a, utils.get_rss(a, self)) for a in active_nodes]
                powers.sort(key=lambda tup: tup[1], reverse=True)
                # only success for the highest power if > power_threshold
                if powers[0][1] >= powers[1][1] + power_threshold:
                    active_node = powers[0][0]
                else:
                    print(f"{self.uid}\tCollision detected")
                    self.collisions.append(self.env.now)
                    active_node = None

            if active_node is not None:
                time_tx_done = active_node.done_tx
                yield self.env.timeout(abs(self.env.now - time_tx_done))
                rx_packet = active_node.message_in_tx
                print(f"{self.uid}\tRx packet from {active_node.uid} {rx_packet}")
                packet_for_us = self.handle_rx_msg(rx_packet)

        return packet_for_us

    def tx(self, route_discovery: bool = False):
        # TODO do CAD before, and schedule TX for next time if channel is not free

        if route_discovery and self.route_discovery_forward_buffer is not None:
            self.message_in_tx = self.route_discovery_forward_buffer
        elif not route_discovery and (len(self.data_buffer) > 0 or len(self.forwarded_mgs_buffer) > 0):
            self.message_in_tx = Message(MessageType.TYPE_ROUTED,
                                         0,
                                         0,
                                         self.best_route,
                                         self.uid,
                                         self.data_buffer,
                                         self.forwarded_mgs_buffer)

        if self.message_in_tx is not None:
            self.state_change(SensorNodeState.STATE_PREAMBLE_TX)

            packet_time = self.message_in_tx.time()
            self.done_tx = self.env.now + settings.PREAMBLE_DURATION_S + packet_time

            yield self.env.timeout(settings.PREAMBLE_DURATION_S)

            self.state_change(SensorNodeState.STATE_TX)
            print(f"{self.uid}\t Sending packet {self.message_in_tx.header.uid} with size: {self.message_in_tx.size()} bytes")

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
        self.state_change(SensorNodeState.STATE_CAD)
        yield self.env.timeout(settings.TIME_CAD_WAKE_S + settings.TIME_CAD_STABILIZE_S)

        # check which nodes are now in PREAMBLE_TX state
        nodes_sending_preamble = self.get_nodes_in_state(SensorNodeState.STATE_PREAMBLE_TX)
        active_nodes = []
        for n in nodes_sending_preamble:
            if utils.in_range(n, self):
                active_nodes.append(n)

        # check after CAD perform, if it was transmitting during the full window
        yield self.env.timeout(settings.TIME_CAD_PERFORM_S)

        cad_detected = False
        for n in active_nodes:
            if n.state is SensorNodeState.STATE_PREAMBLE_TX:
                # OK considered TX and we need to listen
                cad_detected = True
                break
        yield self.env.timeout(settings.TIME_CAD_PROC_S)
        return cad_detected

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

        axis.plot(state_time, states)
        axis.scatter(self.collisions, [0]*len(self.collisions), edgecolor="green")
        axis.grid()

        if plot_labels:

            states = sorted(self.states, key=lambda tup: tup.value)
            y = [s.value for s in states]
            labels = [f"{s.fullname}" for s in states]

            axis.set_yticks(y, labels)
        if axis is None:
            plt.show()


    def full_buffer(self):
        return len(self.data_buffer) > settings.MAX_BUF_SIZE_BYTE

    def get_nodes_in_state(self, state):
        nodes = []
        for n in self.nodes:
            if n is not self and n.state is state:
                nodes.append(n)
        return nodes

    def handle_rx_msg(self, rx_packet):
        packet_for_us = False
        update_beacon = False
        if rx_packet.header.uid in self.seen_packets:
            print(
                f"{self.uid}\tPacket already processed {rx_packet.header.uid}")
        else:
            if rx_packet.is_route_discovery():
                packet_for_us = True
                self.route_discovery_forward_buffer = copy(rx_packet)
                self.tx_collision_timer.start(restart=True)
            elif rx_packet.is_routed() and rx_packet.header.address == self.uid:
                packet_for_us = True
                print(f"{self.uid}\tIt's for us to forward")
                # update tx timer
                self.tx_aggregation_timer.step_up()
                self.forwarded_mgs_buffer.append(rx_packet)
                self.tx_aggregation_timer.start(restart=True)  # restart timer with new back-off

            self.seen_packets.append(rx_packet.header.uid)

        return packet_for_us



