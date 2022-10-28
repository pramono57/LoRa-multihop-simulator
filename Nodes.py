import utils
from utils import random

import numpy as np
import simpy
from aenum import Enum, MultiValue, auto, IntEnum

from config import settings

from utils import random
from Timers import TxTimer, TimerType

from Packets import DataPacket, BeaconPacket
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
    STATE_BEACON = auto()
    STATE_RX = auto()
    STATE_PROC = auto()


class Gateway:
    def __init__(self, env: simpy.Environment, _id):
        self._id = _id
        self._env = env
        self.beacon = BeaconPacket(0, 0)
        self.packet_in_tx = None
        self._state = SensorNodeState.STATE_INIT
        self.done_tx = None
        self.position = Position(0, 0)
        self.states_time = []
        self.states = []

    def state_change(self, state_to):
        if self._state is None:
            print(f"GW\tState change: None->{state_to.fullname}")
        else:
            print(f"GW\tState change: {self._state.fullname}->{state_to.fullname}")
        if state_to is not self._state:
            self._state = state_to
            self.states.append(state_to)
            self.states_time.append(self._env.now)

    def run(self):
        yield self._env.timeout(2*settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        print(f"{self._id}\tStarting GW {self._id}")

        while True:
            #TODO include CAD before Tx
            self.beacon.id += 1

            self.state_change(SensorNodeState.STATE_PREAMBLE_TX)
            self.packet_in_tx = self.beacon

            packet_time = self.packet_in_tx.time()
            self.done_tx = self._env.now + settings.PREAMBLE_DURATION_S + packet_time

            yield self._env.timeout(settings.PREAMBLE_DURATION_S)

            self.state_change(SensorNodeState.STATE_TX)
            yield self._env.timeout(packet_time)

            self.state_change(SensorNodeState.STATE_SLEEP)
            yield self._env.timeout(settings.GW_BEACON_INTERVAL_S)
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

        self.done_tx = 0
        self.states_time = []
        self.states = []
        self.packet_in_tx = None

        self._id = _id
        self._state = None
        self._env = env
        self._energy_mJ = 0
        self.time_to_sense = None

        self.shortest_path_dst = None
        self.shortest_path_num_hops = None
        self.shortest_path_SNR = None
        self.best_beacon = None
        self.beacon_seen_id = -1  # holds the highest beacon counter, seen (to ensure no duplicates)

        self.tx_beacon_timer = TxTimer(env, TimerType.BEACON)
        self.tx_data_timer = TxTimer(env, TimerType.DATA)
        self.sense_timer = TxTimer(env, TimerType.SENSE)

        self.nodes = []  # list containing the other nodes in the network

        self.position = Position.random(size=100)
        self.dst_node = None  # needs to be populated through routing protocol

    def add_nodes(self, nodes):
        self.nodes = nodes

    def state_change(self, state_to):
        if state_to is self._state and state_to is not SensorNodeState.STATE_SLEEP:
            print("mmm not possible, only sleepy can do this")
        if self._state is None:
            print(f"{self._id}\tState change: None->{state_to.fullname}")
        else:
            print(f"{self._id}\tState change: {self._state.fullname}->{state_to.fullname}")
        if state_to is not self._state:
            self._state = state_to
            self.states.append(state_to)
            self.states_time.append(self._env.now)

    def run(self):
        random_wait = np.random.uniform(0, settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        yield self._env.timeout(random_wait)
        print(f"{self._id}\tStarting node {self._id}")
        self.sense_timer.start()

        self.state_change(SensorNodeState.STATE_INIT)
        self._env.process(self.periodic_wakeup())

    def check_sensing(self):
        if self.sense_timer.is_expired():
            self.state_change(SensorNodeState.STATE_SENSING)
            yield self._env.timeout(settings.MEASURE_DURATION_S)
            # schedule timer for transmit
            self.tx_data_timer.start(restart=False)
            self.sense_timer.start()
            self.data_buffer.extend([0, 0])  # populate with 0s for now

    def check_transmit(self):
        # BEACONS are sent separately

        # if full buffer or one of the two timers is expired
        if self.tx_beacon_timer.is_expired():
            yield self._env.process(self.tx(beacon=True))
            self.tx_beacon_timer.reset()
            #

        # elif to ensure beacon TX is not directly followed by a data TX
        # TODO ensure beacon TX does not throttle data TX
        elif self.tx_data_timer.is_expired() or self.full_buffer():
            yield self._env.process(self.tx())
            self.tx_data_timer.reset()

    def periodic_wakeup(self):
        while True:
            self.state_change(SensorNodeState.STATE_SLEEP)
            cad_interval = random(settings.CAD_INTERVAL_RANDOM_S)
            yield self._env.timeout(cad_interval)

            cad_detected = yield self._env.process(self.cad())

            if cad_detected:
                packet_for_us = yield self._env.process(self.receiving())
                self.tx_data_timer.start(restart=True) # restart timer with new back-off
                self.tx_beacon_timer.start(restart=True)
            else:
                yield self._env.process(self.check_transmit())

            yield self._env.process(self.check_sensing())

    def receiving(self):
        """
        After activity during CAD in periodic_cad, we will listen to incoming packets from other nodes

        :return:
        """
        packet_for_us = False
        active_node = None
        print(f"{self._id}\tChecking for RX packet")
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
                    print(f"{self._id}\tCollision detected")
                    self.collisions.append(self._env.now)
                    active_node = None

            if active_node is not None:
                time_tx_done = active_node.done_tx
                yield self._env.timeout(abs(self._env.now - time_tx_done))
                rx_packet = active_node.packet_in_tx
                print(f"{self._id}\tRx packet from {active_node._id} {rx_packet}")
                packet_for_us = self.handle_rx_msg(rx_packet)

        return packet_for_us

    def tx(self, beacon: bool = False):
        # TODO do CAD before, and schedule TX for next time if channel is not free
        # TODO empty buffers again
        self.state_change(SensorNodeState.STATE_PREAMBLE_TX)

        if beacon:
            if self.best_beacon is not None:
                beacon_fwd = copy(self.best_beacon)
                beacon_fwd.num_hops += 1
                beacon_fwd.src = self._id
                self.packet_in_tx = beacon_fwd
            else:
                print("I dont yet have a beacon received")

        else:
            # build data packet
            self.packet_in_tx = DataPacket(self._id, self.dst_node, self.forwarded_mgs_buffer, self.data_buffer)

        if self.packet_in_tx is not None:
            packet_time = self.packet_in_tx.time()
            self.done_tx = self._env.now + settings.PREAMBLE_DURATION_S + packet_time

            yield self._env.timeout(settings.PREAMBLE_DURATION_S)

            self.state_change(SensorNodeState.STATE_TX)
            print(f"{self._id}\t Sending packet with size: {self.packet_in_tx.size()} bytes")

            yield self._env.timeout(packet_time)
            self.done_tx = None
            self.packet_in_tx = None
            self.forwarded_mgs_buffer = []
            self.data_buffer = []

    def cad(self):
        """
        In the CAD state, the node listens for channel activity
        In the beginning it needs to wake-up and stabilise
        After that all messages sent in the CAD window, will be considered received (if power level is above sensitivity)

        Depending on CAD success the node enters RX state or sleep state
        """
        self.state_change(SensorNodeState.STATE_CAD)
        yield self._env.timeout(settings.TIME_CAD_WAKE_S + settings.TIME_CAD_STABILIZE_S)

        # check which nodes are now in PREAMBLE_TX state
        nodes_sending_preamble = self.get_nodes_in_state(SensorNodeState.STATE_PREAMBLE_TX)
        active_nodes = []
        for n in nodes_sending_preamble:
            if utils.in_range(n, self):
                active_nodes.append(n)

        # check after CAD perform, if it was transmitting during the full window
        yield self._env.timeout(settings.TIME_CAD_PERFORM_S)

        cad_detected = False
        for n in active_nodes:
            if n._state is SensorNodeState.STATE_PREAMBLE_TX:
                # OK considered TX and we need to listen
                cad_detected = True
                break
        yield self._env.timeout(settings.TIME_CAD_PROC_S)
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
            if n is not self and n._state is state:
                nodes.append(n)
        return nodes

    def handle_rx_msg(self, rx_packet):
        packet_for_us = False
        update_beacon = False
        if rx_packet.is_beacon():
            packet_for_us = True
            if self.best_beacon is None:
                # first beacon we see, hurray
                update_beacon = True
                self.beacon_seen_id = rx_packet.id
            elif rx_packet.id > self.beacon_seen_id:
                # new beacon!
                self.beacon_seen_id = rx_packet.id
                if rx_packet.num_hops < self.best_beacon.num_hops:
                    update_beacon = True
                # TODO if same, look at SNR
            else:
                pass  # discard msg

            if update_beacon:
                self.best_beacon = rx_packet
                self.shortest_path_dst = self.best_beacon.src
                self.shortest_path_num_hops = self.best_beacon.num_hops
                self.tx_beacon_timer.start()
                print(
                    f"{self._id}\tUpdating routing, new dst {self.shortest_path_dst} with {self.shortest_path_num_hops} hops")

        elif rx_packet.header.dst == self._id and rx_packet.is_data():
            packet_for_us = True
            print(f"{self._id}\tIt's for us to forward")
            # update tx timer
            self.tx_data_timer.step_up()
            self.forwarded_mgs_buffer.append(rx_packet)
        return packet_for_us



