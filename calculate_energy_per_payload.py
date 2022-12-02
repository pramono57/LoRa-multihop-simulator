import pandas as pd

from multihop.Packets import *
from multihop.config import settings
from multihop.Nodes import power_of_state, NodeState
from multihop.preambles import preambles

payload_sizes = range(1, 30, 5)
payload_ns = range(1, 25, 1)

_payload_sizes = []
_payload_ns = []
energies = []
energies_per_payload = []

for payload_n in payload_ns:
    for payload_size in payload_sizes:
        p = Message(MessageType.TYPE_ROUTED, 0, 0, 0, 1, [1] * payload_size, None,
                    [Message(MessageType.TYPE_ROUTED, 0, 0, 0, 1, [1] * payload_size, None, [])] * payload_n)
        print(p)

        if p.size() < 255:
            energy = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S] * power_of_state(NodeState.STATE_PREAMBLE_TX) + p.time() * power_of_state(NodeState.STATE_TX)
            _payload_sizes.append(payload_size)
            _payload_ns.append(payload_n)
            energies.append(energy)
            energies_per_payload.append(energy/(payload_n+1))  # + 1 because of own payload

df = pd.DataFrame({
    "payload_size": _payload_sizes,
    "payload_n": _payload_ns,
    "energy": energies,
    "energy_per_payload": energies_per_payload
})

_df = df.pivot(index='payload_n', columns='payload_size', values='energy_per_payload')
_df.plot()

print("The end")
