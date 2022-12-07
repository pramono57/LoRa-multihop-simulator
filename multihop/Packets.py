import math
import random

from aenum import auto, IntEnum

from .config import settings


class MessageType(IntEnum):
    TYPE_ROUTE_DISCOVERY = auto()
    TYPE_ROUTED = auto()


def time_on_air(num_bytes):
    t_sym = (2.0 ** settings.LORA_SF) / settings.LORA_BANDWIDTH
    payload_symb_n_b = 8 + max(
        math.ceil(
            (
                    8.0 * num_bytes - 4.0 * settings.LORA_SF + 28 + 16 * settings.LORA_CRC - 20 * settings.LORA_IMPLICIT_HEADER) / (
                    4.0 * (settings.LORA_SF - 2 * settings.LORA_LOW_DATA_RATE_OPTIMIZE)))
        * (settings.LORA_CODE_RATE + 4), 0)
    return payload_symb_n_b * t_sym / 1000  # to convert from ms to s


class MessageHeader:
    def __init__(self, msg_type: MessageType, hops, lqi, address):
        self.uid = random.randint(0, 65536)  # 2 byte
        self.type = msg_type                # 1 byte
        self.hops = hops                    # 1 byte
        self.cumulative_lqi = lqi           # 2 bytes
        self.address = address              # 1 byte, determined via the beacons

    @staticmethod
    def size():
        # msg_id 2 + type 1 + hops 1 + lqi 2 + address 1  = 6
        return 7

    def hop(self):
        self.hops += 1

    def __str__(self):
        return f"uid:{self.uid} | type:{self.type} | hops:{self.hops} | " \
               f"lqi: {round(self.cumulative_lqi,2)} | address: {self.address}"


class MessagePayloadChunk:
    def __init__(self, src, data, src_node):
        self.src = src
        self.data = data
        self.len = len(data)

        # Meta data
        self.sent_at = 0
        if src_node is not None:
            self.sent_at = src_node.env.now
        self.arrived_at = 0
        self.hops = 0
        self.trace = [src]
        self.src_node = src_node

        self.collided = False
        self.collided_at = None

    def size(self):
        return self.len + 2  # + 2 header info bytes

    def hop(self, node):
        self.hops += 1
        self.trace.append(node.uid)

    def arrived_at_gateway(self):
        self.arrived_at = self.src_node.env.now
        if len(self.data) > 0:
            self.src_node.arrived_at_gateway(self)

    def handle_collision(self):
        self.collided_at = self.src_node.env.now
        self.collided = True
        self.src_node.collided(self)

    def in_trace(self, uid):
        if self.trace.count(uid) > 1:
            return True
        else:
            return False

    def __str__(self):
        data = ":".join("{:02x}".format(c) for c in self.data)
        return f"(trace:{self.trace} | data:{data})"


class MessagePayload:
    def __init__(self, src, own_data, src_node, forwarded_msgs):
        self.own_data = MessagePayloadChunk(src, own_data, src_node)
        self.forwarded_data = []

        for f in forwarded_msgs:
            self.forwarded_data.extend(f.payload.forwarded_data)
            if f.payload.own_data.len > 0:
                self.forwarded_data.append(f.payload.own_data)

    def size(self):
        size = self.own_data.size()
        for p in self.forwarded_data:
            size += p.size()
        return size

    def set_own_data(self, src, own_data, src_node):
        self.own_data = MessagePayloadChunk(src, own_data, src_node)

    def arrived_at_gateway(self):
        if self.own_data.len > 0:
            self.own_data.arrived_at_gateway()
        for p in self.forwarded_data:
            p.arrived_at_gateway()

    def handle_collision(self):
        if self.own_data.len > 0:
            self.own_data.handle_collision()
        for p in self.forwarded_data:
            p.handle_collision()

    def in_trace(self, uid):
        for p in self.forwarded_data:
            if p.in_trace(uid):
                return True
    def __str__(self):
        str = f"own_data:{self.own_data} | forwarded_data:["
        for f in self.forwarded_data:
           str = f"{str} {f} |"
        if len(self.forwarded_data) > 0:
            return f"{str[:-1]}]"
        else:
            return f"{str}]"


class Message:
    """
    Structure
    [msg_id   type   dst   src   len   own_data    [src   len   data] [src   len   data] [src   len   data] ...]
    [            HEADER             ][ OWN DATA ]  [                       FORWARDED MSGS                      ]
    """

    def __init__(self, msg_type: MessageType, hops, lqi, dst, src, own_data, src_node, forwarded_msgs):
        self.header = MessageHeader(msg_type, hops, lqi, dst)
        self.payload = MessagePayload(src, own_data, src_node, forwarded_msgs)

        # Meta
        self.collided = False
    def size(self):
        size = 0
        size += self.header.size()
        size += self.payload.size()
        return size

    def hop(self, node):
        self.header.hop()
        if self.is_route_discovery():
            self.payload.own_data.hop(node)
        for p in self.payload.forwarded_data:
            p.hop(node)

    def copy(self):
        cpy = Message(self.header.type, self.header.hops, self.header.cumulative_lqi, self.header.address,
                      self.payload.own_data.src, self.payload.own_data.data, self.payload.own_data.src_node, self.payload.forwarded_data)
        cpy.payload.own_data.trace = self.payload.own_data.trace.copy()
        cpy.header.uid = self.header.uid
        return cpy

    def time(self):
        # return the time on air
        return time_on_air(self.size())

    def is_route_discovery(self):
        return self.header.type is MessageType.TYPE_ROUTE_DISCOVERY

    def is_routed(self):
        return self.header.type is MessageType.TYPE_ROUTED

    def arrived_at_gateway(self):
        self.payload.arrived_at_gateway()

    def handle_collision(self):
        self.collided = True
        self.payload.handle_collision()

    def in_trace(self, uid):
        return self.payload.in_trace(uid)

    def __str__(self):
        return f"{self.header} || {self.payload}"