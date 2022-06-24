import math
from aenum import Enum, MultiValue, auto, IntEnum

import random
import simpy
from config import settings
import numpy as np


class Position:
    def __init__(self, x, y):
        self.x = 0
        self.y = 0


class PacketType(IntEnum):
    TYPE_BEACON = auto()
    TYPE_UPLINK_MESSAGE = auto()


class PacketHeader:
    def __init__(self, msg_type: PacketType, src, dst):
        self.id = random.randint(0, 255)
        self.type = msg_type
        self.src = src
        self.dst = dst  # determined via the beacons
        self.num_hops = 0

    @staticmethod
    def size():
        # msg_id + type + hops + src + dst + len = 6
        return 6


def time_on_air(num_bytes):
    t_sym = (2.0 ** settings.SF) / settings.BW
    payload_symb_n_b = 8 + max(
        math.ceil(
            (
                    8.0 * num_bytes - 4.0 * settings.SF + 28 + 16 * settings.CRC - 20 * settings.IH) / (
                    4.0 * (settings.SF - 2 * settings.DE)))
        * (settings.CR + 4), 0)
    return payload_symb_n_b * t_sym * 1000  # to covnert from ms to s


class Packet:
    def __init__(self, msg_type: PacketType, src, dst, own_len, forwarded_packets=None):
        self.header = PacketHeader(msg_type, src, dst)
        self.own_len = own_len  # own payload data length
        self.forwarded_payload = forwarded_packets  # payload forwarded nodes

    def add_hop(self):
        self.header.num_hops += 1

    def size(self):
        size = self.own_len
        for p in self.forwarded_payload:
            size += p.own_len + 2  # add two extra bytes for the src_id and len
        return size

    def time(self):
        # return the time on air

        num_bytes = PacketHeader.size()
        num_bytes += self.size()

        return time_on_air(num_bytes)


class GatewayState(IntEnum):
    STATE_INIT = auto()
    STATE_BEACON = auto()
    STATE_RX = auto()
    STATE_PROC = auto()


class Gateway:
    # TODO ensure Singleton?
    def __init__(self):
        pass


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


class SensorNode:
    def __init__(self, env: simpy.Environment, _id):
        self.done_tx = 0
        self.states_time = []
        self.states = []
        self.packet = None
        self.forwarded_packets = []
        self._id = _id
        self._state = None
        self._env = env
        self._energy_mJ = 0

        self.position = Position(0, 0)  # TODO
        self._data_len = 0  # total data to sent
        self._data_to_forward = []
        self.dst_node = None  # needs to be populated through routing protocol
        self.radio_resource = simpy.Resource(env, capacity=1)

    def add_nodes(self, nodes):
        self.nodes = nodes

    def state_change(self, state_to: SensorNodeState):
        if state_to is self._state and state_to is not SensorNodeState.STATE_SLEEP:
            print("mmm not possible, only sleepy can do this")
        if self._state is None:
            print(f"Node {self._id} State change: None->{state_to.fullname}")
        else:
            print(f"Node {self._id} State change: {self._state.fullname}->{state_to.fullname}")
        self._state = state_to
        self.states.append(state_to)
        self.states_time.append(self._env.now)

    def run(self):
        random_wait = np.random.uniform(0, settings.MAX_DELAY_START_PER_NODE_RANDOM_S)
        yield self._env.timeout(random_wait)
        print(f"Starting node {self._id}")

        self.state_change(SensorNodeState.STATE_INIT)

        self._env.process(self.sensing())
        self._env.process(self.periodic_cad())

    def sensing(self):
        while True:
            yield self._env.timeout(settings.MEASURE_INTERVAL_S)

            self.state_change(SensorNodeState.STATE_SENSING)
            yield self._env.timeout(settings.MEASURE_DURATION_S)
            self._data_len += 2
            self._start_tx_intent = True  # TODO
            # enter sleepy-mode
            self.state_change(SensorNodeState.STATE_SLEEP)

            # enter out of sleepy mode
            if self._data_len > settings.MAX_BUF_SIZE_BYTE:
                # schedule a transmit
                self._env.process(self.transmit())

    def periodic_cad(self):
        while True:
            # TODO check if not in colliding state

            cad_detected = yield self._env.process(self.cad())
            if cad_detected:
                yield self._env.process(self.receiving())
            # enter sleepy-mode
            self.state_change(SensorNodeState.STATE_SLEEP)
            cad_interval = self.random(settings.CAD_INTERVAL_RANDOM_S)
            yield self._env.timeout(cad_interval)
            # enter out of sleepy mode

    def receiving(self):
        """
        After activity during CAD in periodic_cad, we will listen to incoming packets from other nodes

        :return:
        """
        req = self.yield_radio()
        print(f"Checking for RX packet")
        self.state_change(SensorNodeState.STATE_RX)
        found_tx = False
        time_tx_done = 0
        active_node = None
        for n in self.nodes:
            if n is not self:
                if n._state is SensorNodeState.STATE_PREAMBLE_TX or SensorNodeState.STATE_TX:
                    # TODO check collision for now, first is ok
                    found_tx = True
                    active_node = n
                    time_tx_done = n.done_tx
                    break

        if found_tx:
            # TODO update energy RX
            yield self._env.timeout(abs(self._env.now - time_tx_done))
        else:
            self.state_change(SensorNodeState.STATE_SLEEP)
            print("Nobody was transmitting.")
        self.release_radio(req)

        rx_packet = active_node.packet
        print(f"Rx packet from {active_node._id} {rx_packet}")
        # TODO handle msg

    def transmit(self):
        # starts the intent of transmitting
        """
        [intent of Tx] ----- back-off ----- [CAD][TX] (no activity during CAD)
        [intent of Tx] ----- back-off ----- [CAD] --- back-off ---- [CAD][Tx] (activity during first CAD)
        """
        back_off = self.random(settings.TX_INTENT_BACKOFF_RANDOM)
        if self._state is not SensorNodeState.STATE_SLEEP:
            self.state_change(SensorNodeState.STATE_SLEEP)
        yield self._env.timeout(back_off)

        # first do CAD, if no AD -> TX immediately, otherwise wait TX_BACKOFF_RANDOM
        cad_detected = True
        iter_cad = 0
        while cad_detected and iter_cad < settings.MAX_ITER_CAD:
            cad_detected = yield self._env.process(self.cad())
            # after CAD we go to sleep
            back_off = self.random(settings.TX_BACKOFF_RANDOM)
            self._energy_mJ += settings.POWER_SLEEP_mW * back_off
            self.state_change(SensorNodeState.STATE_SLEEP)
            yield self._env.timeout(back_off)
            iter_cad += 1

        # in this state we send all accumulated packets
        req = self.yield_radio()

        # start with sending the preamble
        self._energy_mJ += settings.POWER_TX_mW * settings.PREAMBLE_DURATION_S
        self.state_change(SensorNodeState.STATE_PREAMBLE_TX)
        yield self._env.timeout(settings.PREAMBLE_DURATION_S)
        # now the real data transmission can take place
        self.build_packet()
        packet_time = self.packet.time()
        self._energy_mJ += settings.POWER_TX_mW * packet_time
        self.state_change(SensorNodeState.STATE_TX)
        self.done_tx = self._env.now + settings.PREAMBLE_DURATION_S + packet_time
        yield self._env.timeout(packet_time)
        self.done_tx = 0
        self.packet = None  # clear packet (hopefully no collisions)
        self.release_radio(req)
        # our work is done! Time for a nap
        self._next_state = SensorNodeState.STATE_SLEEP

    # def processing(self):
    #     self._state = SensorNodeState.STATE_PROC
    #     yield self._env.timeout(0)  # TODO

    def cad(self):
        """
        In the CAD state, the node listens for channel activity
        In the beginning it needs to wake-up and stabilise
        After that all messages sent in the CAD window, will be considered received (if power level is above sensitivity)

        Depending on CAD success the node enters RX state or sleep state
        """
        req = self.yield_radio()
        self.state_change(SensorNodeState.STATE_CAD)

        self._energy_mJ += settings.ENERGY_CAD_CYCLE_mJ
          # SPLIT IN TWO
        yield self._env.timeout(settings.TIME_CAD_WAKE_S + settings.TIME_CAD_STABILIZE_S)

        # check which nodes are now in PREAMBLE_TX state
        nodes_sending_preamble = []
        for n in self.nodes:
            if n is not self and n._state is SensorNodeState.STATE_PREAMBLE_TX and in_range(n, self):
                nodes_sending_preamble.append(n)

        # check after CAD perform, if it was transmitting during the full window
        yield self._env.timeout(settings.TIME_CAD_PERFORM_S)

        cad_detected = False
        for n in self.nodes:
            if n is not self and n._state is SensorNodeState.STATE_PREAMBLE_TX and in_range(n,
                                                                                            self) and n in nodes_sending_preamble:
                # OK considered TX and we need to listen
                cad_detected = True
                break
        yield self._env.timeout(settings.TIME_CAD_PROC_S)
        self.release_radio(req)
        return cad_detected

    def build_packet(self):
        self.packet = Packet(PacketType.TYPE_UPLINK_MESSAGE, self._id, self.dst_node, self._data_len,
                             self.forwarded_packets)

        # clear packets
        self.forwarded_packets = []
        self._data_len = 0

    def yield_radio(self):
        req = self.radio_resource.request()
        yield req
        return req

    def release_radio(self, request):
        self.radio_resource.release(request)

    def plot_states(self):
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

        plt.figure()
        plt.plot(state_time, states)
        plt.show()

    def random(self, min_max: tuple):
        return random.uniform(*min_max)


def distance(p1, p2):
    return np.sqrt(np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y))


def in_range(n1: SensorNode, n2: SensorNode):
    # TODO make better
    return True
    # return distance(n1.position, n2.position) < 100
