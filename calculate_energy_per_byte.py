# Simulate max distance between two nodes
import pandas as pd
import os

from multihop.Packets import *
from multihop.config import settings
from multihop.Nodes import power_of_state, NodeState
from multihop.preambles import preambles

sizes = range(1, 255 - 7, 1)
energies = []
energies_per_byte = []

for size in sizes:
    p = Message(settings, MessageType.TYPE_ROUTED, 0, 0, 0, 1, [1] * size, 0, None, [])
    energy = preambles[settings.LORA_SF][settings.MEASURE_INTERVAL_S] * power_of_state(settings, NodeState.STATE_PREAMBLE_TX) + p.time() * power_of_state(settings, NodeState.STATE_TX)
    energies.append(energy)
    energies_per_byte.append(energy/size)

df = pd.DataFrame({
    "size": sizes,
    "energy": energies,
    "energy_per_byte": energies_per_byte
})

os.makedirs("results", exist_ok=True)
filename = "./results/energy_per_byte_calculation.csv"
df.to_csv(filename, index=False)

print("The end")