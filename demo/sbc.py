import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sbc import SimpleBinaryClassification
from brian2 import ms, start_scope, defaultclock, Network
import matplotlib.pyplot as plt

start_scope()
defaultclock.dt = 0.01*ms

model = SimpleBinaryClassification()

# Define input spikes
model.input.set_spikes(
    indices=[0,0],
    times=[0,5.5]*ms
)

net = Network(*model.get_objects())
net.run(50*ms)

print("Input spikes:", model.spikes_in.t)
print("Output spikes:", model.spikes_out.t)
print("Final weight:", model.syn.w[:])

plt.figure(figsize=(8, 4))
plt.plot(model.vmon.t/ms, model.vmon.v[0])
plt.axhline(1, linestyle='--', color='r', label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v')
plt.title('Output neuron membrane potential')
plt.legend()
plt.show()