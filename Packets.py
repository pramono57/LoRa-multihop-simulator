import math
import random

from aenum import auto, IntEnum

from config import settings


class PacketType(IntEnum):
    TYPE_BEACON = auto()
    TYPE_DATA = auto()


def time_on_air(num_bytes):
    t_sym = (2.0 ** settings.SF) / settings.BW
    payload_symb_n_b = 8 + max(
        math.ceil(
            (
                    8.0 * num_bytes - 4.0 * settings.SF + 28 + 16 * settings.CRC - 20 * settings.IH) / (
                    4.0 * (settings.SF - 2 * settings.DE)))
        * (settings.CR + 4), 0)
    return payload_symb_n_b * t_sym / 1000  # to convert from ms to s


class DataPacketHeader:
    def __init__(self, msg_type: PacketType, src, dst):
        self.id = random.randint(0, 255)
        self.type = msg_type
        self.src = src
        self.dst = dst  # determined via the beacons

    @staticmethod
    def size():
        # msg_id + type + hops + src + dst + len = 6
        return 6


class ForwardMessage:
    def __init__(self, src, data):
        self.src = src
        self.data = data
        self.len = len(data)

    def size(self):
        return self.len + 2  # + 2 header info bytes


class DataPacket:
    """
    Structure
    [msg_id   type   dst   src   len   own_data    [src   len   data] [src   len   data] [src   len   data] ...]
    [            HEADER             ][ OWN DATA ]  [                       FORWARDED MSGS                      ]
    """

    def __init__(self, src, dst, forwarded_msgs, data_buffer):
        self.header = DataPacketHeader(PacketType.TYPE_DATA, src, dst)
        self.own_data = data_buffer
        self.forward_msgs = []
        for f in forwarded_msgs:
            self.forward_msgs.extend(f.forward_msgs)
            self.forward_msgs.append(ForwardMessage(f.header.src, f.own_data))

    def size(self):
        size = 0
        size += DataPacketHeader.size()
        size += len(self.own_data)
        for p in self.forward_msgs:
            size += p.size()
        return size

    def time(self):
        # return the time on air
        return time_on_air(self.size())

    def is_beacon(self):
        return False

    def is_data(self):
        return True


class BeaconPacket:
    """
    Structure
    [msg_id   type   hops   src]
    """

    def __init__(self, id, src ):
        self.id = id
        self.src = src
        self.num_hops = 0

    def size(self):
        return 4

    def time(self):
        # return the time on air
        return time_on_air(self.size())

    def is_beacon(self):
        return True

    def is_data(self):
        return False

