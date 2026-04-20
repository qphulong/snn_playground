"""
inference_simple.py
===================
Runs the trained SNN on inference audio files and computes pairwise
similarity using the spikernel, but only over a selected subset of neurons.
"""

import os, sys, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from brian2 import *
from src.utils.spike_encoding import compute_spike_input_current
from src.spikernel.spikernel import spikernel_normalized

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_epoch_031.npz")
INFER_DIR    = "datasets/vox1_single_person_nano_2/test"
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vizs")

BIN_MS = 100
N_MAX  = 5
LAM    = 0.99
MU     = 0.35
Q      = 3.0

# ── Choose which neurons to include ──────────────────────────────────────────
NEURON_INDICES = []   # ← edit this list freely
# ─────────────────────────────────────────────────────────────────────────────

# Network hyperparameters (must match training)
N_IN = 700
N_H  = 700
DT_SIM = 1 * ms

tau_m       = 50 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 0.15
v_th_in     = 1.0

tau_h    = 50 * ms
tau_vth  = 200 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading weights: {WEIGHTS_PATH}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
data     = np.load(WEIGHTS_PATH, allow_pickle=True)
w_matrix = data["final_weights_matrix"].astype(np.float64)
print(f"  shape: {w_matrix.shape}  range: [{w_matrix.min():.4f}, {w_matrix.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# Collect audio files
# ─────────────────────────────────────────────────────────────────────────────

wav_files = sorted(glob.glob(os.path.join(INFER_DIR, "**", "*.wav"), recursive=True))
print(f"\nFound {len(wav_files)} wav files")
if not wav_files:
    raise RuntimeError(f"No .wav files found under {INFER_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Network runner
# ─────────────────────────────────────────────────────────────────────────────

def run_network(audio_path, w_matrix):
    I, T = compute_spike_input_current(
        audio_path, scale=1,
        sustained_per_band=4, onset_per_band=2, phase_per_band=1,
        sust_spread_min=0.7, sust_spread_max=1.3,
    )

    start_scope()
    defaultclock.dt = DT_SIM
    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in, threshold="v > v_th_in",
                       reset="v=0; a+=beta", refractory=2*ms, method="euler")

    eqs_h = f"""
    dv/dt   = -v / tau_h                      : 1
    dvth/dt = -(vth - {vth_rest}) / tau_vth   : 1
    """
    G_h = NeuronGroup(N_H, eqs_h, threshold="v > vth",
                      reset=f"v=0; vth=vth+{vth_jump};",
                      refractory=2*ms, method="euler")
    G_h.vth = vth_init

    S = Synapses(G_in, G_h, "w : 1", on_pre="v_post += w")
    S.connect()
    S.w = w_matrix[np.array(S.i), np.array(S.j)]

    lat = Synapses(G_h, G_h, on_pre="v_post = clip(v_post * 0.8, 0, inf)")
    lat.connect(condition="i != j")

    mon = SpikeMonitor(G_h)
    run(T * DT_SIM)

    spike_i  = np.array(mon.i,      dtype=np.int32)
    spike_t  = np.array(mon.t / ms, dtype=np.float32)
    duration_ms = float(T) * float(DT_SIM / ms)
    return spike_i, spike_t, duration_ms


def spikes_to_binned_sequence(spike_i, spike_t_ms, neuron_indices, duration_ms, bin_ms):
    """
    Returns a (len(neuron_indices), n_bins) spike-count array,
    keeping only the neurons listed in neuron_indices.
    """
    n_bins   = max(1, int(np.ceil(duration_ms / bin_ms)))
    idx_set  = set(neuron_indices)
    idx_map  = {nid: pos for pos, nid in enumerate(neuron_indices)}

    seq = np.zeros((len(neuron_indices), n_bins), dtype=np.float32)
    if len(spike_i) == 0:
        return seq

    bin_idx = np.clip(np.floor(spike_t_ms / bin_ms).astype(np.int32), 0, n_bins - 1)
    for ni, bi in zip(spike_i, bin_idx):
        if ni in idx_set:
            seq[idx_map[ni], bi] += 1
    return seq

# ─────────────────────────────────────────────────────────────────────────────
# Run inference
# ─────────────────────────────────────────────────────────────────────────────

if not NEURON_INDICES:
    NEURON_INDICES = list(range(N_H))

print(f"\nNeuron indices used ({len(NEURON_INDICES)}): {NEURON_INDICES}")
print("=" * 60)

sequences, labels = [], []

for idx, path in enumerate(wav_files):
    rel = os.path.relpath(path, start=os.path.dirname(INFER_DIR))
    print(f"\n[{idx+1}/{len(wav_files)}] {rel}")

    spike_i, spike_t, dur_ms = run_network(path, w_matrix)
    seq = spikes_to_binned_sequence(spike_i, spike_t, NEURON_INDICES, dur_ms, BIN_MS)
    sequences.append(seq)

    parts = rel.replace("\\", "/").split("/")
    labels.append("/".join(parts[-2:]) if len(parts) >= 2 else rel)

    print(f"  spikes={len(spike_i):,}  duration={dur_ms:.0f}ms  bins={seq.shape[1]}")
    print(f"  spike sequence array (shape {seq.shape}):")
    print(seq)

# ─────────────────────────────────────────────────────────────────────────────
# Pairwise spikernel similarity
# ─────────────────────────────────────────────────────────────────────────────

n = len(sequences)
print(f"\nComputing {n}×{n} normalized spikernel matrix …")

K_norm = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    for j in range(i, n):
        val = spikernel_normalized(sequences[i], sequences[j], N_MAX, LAM, MU, Q)
        K_norm[i, j] = K_norm[j, i] = val

print("\nNormalised similarity matrix K_norm:")
print(K_norm)
print("\nLabels:", labels)

# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap(matrix, labels, title, save_path):
    n = len(labels)
    fig_size = max(7, n * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised similarity")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 10 - n // 4))
    ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 4))

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(5, 9 - n // 4),
                    color="white" if val < 0.55 else "black")

    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="red",
                                   linewidth=1.5, zorder=3))

    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("Audio file")
    ax.set_ylabel("Audio file")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {save_path}")


heatmap_path = os.path.join(OUTPUT_DIR, "spikernel_similarity.png")
make_heatmap(
    K_norm, labels,
    title=f"Spikernel similarity  (neurons={NEURON_INDICES[-5:]}, N_MAX={N_MAX}, λ={LAM}, μ={MU}, bin={BIN_MS}ms)",
    save_path=heatmap_path,
)