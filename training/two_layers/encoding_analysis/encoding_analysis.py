"""
Encoding Analysis - Input Layer Spike Similarity

Processes audio through input layer, computes accumulated spike counts,
and calculates pairwise similarity using cosine similarity.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import *
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current

# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_DIR = "datasets/slicing_window_analysis/sample1"
N_SAMPLES = 10
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Input Layer Hyperparameters
# ============================================================

N_IN = 700
DT_SIM = 1 * ms
tau_m       = 40 * ms
tau_a       = 20 * ms
tau_current = 1 * ms
beta        = 0.25
v_th_in     = 1.0

# ============================================================
# Find audio files
# ============================================================

wav_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.wav")))[:N_SAMPLES]
if not wav_files:
    print(f"ERROR: No .wav files found in {DATASET_DIR}")
    sys.exit(1)

print(f"Processing {len(wav_files)} samples")

# ============================================================
# Helper functions
# ============================================================

def spikes_to_accumulated_counts(spike_i, n_neurons):
    """Convert spike trains to accumulated spike count vector (n_neurons,)."""
    counts = np.zeros(n_neurons, dtype=np.float32)
    if len(spike_i) == 0:
        return counts
    for ni in spike_i:
        counts[ni] += 1
    return counts


# ============================================================
# Process samples - compute spike-count matrices
# ============================================================

all_sequences = []
sample_names = []

print(f"\n{'='*60}\nProcessing samples\n{'='*60}\n")

for sample_idx, audio_path in enumerate(wav_files):
    rel = os.path.relpath(audio_path, start=os.path.dirname(DATASET_DIR))
    print(f"[{sample_idx+1}/{len(wav_files)}] {rel}")

    try:
        I, T = compute_spike_input_current(
            audio_path, scale=1,
            sustained_per_band=4, onset_per_band=2, phase_per_band=1,
            sust_spread_min=0.7, sust_spread_max=1.3
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    duration_ms = float(T) * float(DT_SIM / ms)

    # Brian2 simulation
    start_scope()
    defaultclock.dt = DT_SIM
    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in, threshold="v > v_th_in",
                       reset="v=0; a+=beta", refractory=2*ms, method="euler")
    mon = SpikeMonitor(G_in)
    run(T * DT_SIM)

    # Extract and accumulate spike counts
    spike_i = np.array(mon.i, dtype=np.int32)
    spike_counts = spikes_to_accumulated_counts(spike_i, N_IN)

    all_sequences.append(spike_counts)

    parts = rel.replace("\\", "/").split("/")
    sample_names.append(parts[-1])

    print(f"  spikes={len(spike_i):,}  shape={spike_counts.shape}")

n = len(all_sequences)
print(f"\nProcessed {n} samples")

# ============================================================
# Compute similarity matrix using cosine similarity
# ============================================================

print(f"\n{'='*60}")
print(f"Computing {n}×{n} cosine similarity matrix")
print(f"{'='*60}\n")

# Stack all sequences into a matrix (n_samples, 700)
sequences_matrix = np.array(all_sequences, dtype=np.float64)
print(f"Sequences matrix shape: {sequences_matrix.shape}")

# Compute cosine similarity
K = cosine_similarity(sequences_matrix)

print(f"\nSimilarity range: [{K.min():.4f}, {K.max():.4f}]")

# ============================================================
# Plot raster plots for all samples
# ============================================================

print("\nPlotting raster plots for all samples...")

# Re-process to get spike times for raster plots
all_spikes = []  # List of (spike_i, spike_t) tuples

for sample_idx, audio_path in enumerate(wav_files):
    try:
        I, T = compute_spike_input_current(
            audio_path, scale=1,
            sustained_per_band=4, onset_per_band=2, phase_per_band=1,
            sust_spread_min=0.7, sust_spread_max=1.3
        )
    except Exception as e:
        print(f"  ERROR processing {audio_path}: {e}")
        continue

    duration_ms = float(T) * float(DT_SIM / ms)

    # Brian2 simulation
    start_scope()
    defaultclock.dt = DT_SIM
    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in, threshold="v > v_th_in",
                       reset="v=0; a+=beta", refractory=2*ms, method="euler")
    mon = SpikeMonitor(G_in)
    run(T * DT_SIM)

    # Extract spike times
    spike_i = np.array(mon.i, dtype=np.int32)
    spike_t = np.array(mon.t / ms, dtype=np.float32)
    all_spikes.append((spike_i, spike_t, duration_ms))

# Plot raster plots
for sample_idx, (spike_i, spike_t, duration_ms) in enumerate(all_spikes):
    fig, ax = plt.subplots(figsize=(12, 8))

    if len(spike_i) > 0:
        ax.scatter(spike_t, spike_i, s=1, alpha=0.5, color="black")

    # Add vertical lines every 1000ms for visibility
    for t in np.arange(0, duration_ms + 1000, 1000):
        ax.axvline(x=t, color="red", linestyle="-", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Neuron Index", fontsize=12)
    ax.set_title(f"Spike Raster - Sample {sample_idx} ({sample_names[sample_idx]})", fontsize=12)
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(-1, N_IN)

    plt.tight_layout()

    raster_path = os.path.join(OUTPUT_DIR, f"raster_sample_{sample_idx:02d}.png")
    plt.savefig(raster_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {raster_path}")
    plt.close()

# ============================================================
# Plot heatmap
# ============================================================

print("\nPlotting heatmap...")

n = len(sample_names)
fig_size = max(7, n * 0.8 + 2)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))

im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(sample_names, rotation=45, ha="right", fontsize=max(6, 10 - n // 4))
ax.set_yticklabels(sample_names, fontsize=max(6, 10 - n // 4))

for i in range(n):
    for j in range(n):
        val = K[i, j]
        txt_color = "white" if val < 0.55 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=max(5, 9 - n // 4), color=txt_color)

ax.set_xticks(np.arange(n) - 0.5, minor=True)
ax.set_yticks(np.arange(n) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=0.5)

for k in range(n):
    ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=1.5, zorder=3))

ax.set_title(f"Input Layer Spike Similarity (Cosine Similarity)", fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "similarity.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"Saved: {heatmap_path}")
plt.close()

print(f"\n{'='*60}\nDone!\n{'='*60}")
