import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sbc import SimpleBinaryClassification
from brian2 import *

# -------------------------------------------------
# Setup
# -------------------------------------------------
start_scope()
defaultclock.dt = 0.01 * ms

model = SimpleBinaryClassification()

# -------------------------------------------------
# Experiment definition
# -------------------------------------------------
patterns = [[0.01, 5.5], [0.01, 8.7],[0.01,5.6],[0.01,8.4]] # sample data
pattern_duration = 20 * ms
epoch_duration = 80 * ms
n_epochs = 10
total_time = epoch_duration * n_epochs

# -------------------------------------------------
# Precompute spikes
# -------------------------------------------------
indices, times = [], []
t = 0 * ms
for _ in range(n_epochs):
    for pat in patterns:
        for s in pat:
            indices.append(0)
            times.append(t + s * ms)
        t += pattern_duration

model.input.set_spikes(indices, times)

# -------------------------------------------------
# History
# -------------------------------------------------
weight_history = []
winner_history = []

prev_weights = model.syn.w[:].copy()

# -------------------------------------------------
# WTA + weight restore
# -------------------------------------------------
# Note: i have found no way making this function run at the end of each timestamp
# currently it only run before each timestamp, so that is out of sync,
# but the fact that it does train prove that it works well for a prototype
# the weights of synapses did converged
def wta_step():
    global prev_weights
            
    t1 = defaultclock.t
    t0 = t1 - pattern_duration

    curr_weights = model.syn.w[:].copy()

    weight_history.append(curr_weights.copy())

    mask = (model.spikes_out.t >= t0) & (model.spikes_out.t <= t1)
    if not np.any(mask):
        winner_history.append(None)
        model.output.v = 0.0
        prev_weights = curr_weights
        return

    spiking_neurons = np.unique(model.spikes_out.i[mask])

    # Choose winner: lowest pre-spike voltage
    v_vals = np.array([model.output.v_pre_spike[i] for i in spiking_neurons])
    winner = int(spiking_neurons[np.argmin(v_vals)])
    winner_history.append(winner)

    # ---- Weight restore ----
    for post_idx in range(model.output.N):
        syn_indices = np.where(model.syn.j[:] == post_idx)[0]

        if post_idx == winner:
            model.syn.w[syn_indices] = curr_weights[syn_indices]
        else:
            model.syn.w[syn_indices] = prev_weights[syn_indices]
            model.output.v[post_idx] = 0.0

    prev_weights = model.syn.w[:].copy()

wta_op = NetworkOperation(wta_step, dt=20*ms, when='end')

# -------------------------------------------------
# Run
# -------------------------------------------------
net = Network(*model.get_objects(), wta_op)
net.run(total_time)

# -------------------------------------------------
# Epoch weight report
# -------------------------------------------------
print("\nWeights after each epoch:")
for e in range(n_epochs-1):
    idx = (e + 1) * 2 - 1
    print(f"Epoch {e+1}: {weight_history[idx]}")

print("\nWinner history (every 20 ms):")
print(winner_history)

# -------------------------------------------------
# Plot neuron 0 membrane potential
# -------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(model.vmon.t/ms, model.vmon.v[0])
plt.axhline(1, linestyle='--', color='r')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential')
plt.title('Neuron 0 membrane potential during training')
plt.tight_layout()
plt.show()