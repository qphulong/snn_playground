"""
Encoding Analysis - Input Layer Spike Similarity

Processes audio through input layer, computes spike-count sequences,
and calculates pairwise similarity using Spikernel.

NEW: Selective neuron filtering via 'neurons' list.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import *
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.spikernel.spikernel import spikernel_normalized

# ============================================================
# USER CONFIGURATION
# ============================================================

DATASET_DIR = "datasets/slicing_window_analysis/sample1"
N_SAMPLES = 10
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== NEW: Selective Neurons ======================
# Put neuron indices you want to keep here.
# Example: neurons = [10, 50, 123, 300, 450, 699]
# If left empty [], all 700 neurons will be used.
neurons = [100,200,300,400,500,600]          # ←←← EDIT THIS ARRAY MANUALLY

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
# Spikernel Hyperparameters
# ============================================================

SPIKERNEL_N_MAX = 5
SPIKERNEL_LAM = 0.99
SPIKERNEL_MU = 0.35
SPIKERNEL_Q = 3
SPIKERNEL_EPS = 1e-12

# ============================================================
# Find audio files
# ============================================================

wav_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.wav")))[:N_SAMPLES]
if not wav_files:
    print(f"ERROR: No .wav files found in {DATASET_DIR}")
    sys.exit(1)

print(f"Processing {len(wav_files)} samples")
if neurons:
    print(f"Using only {len(neurons)} selected neurons: {neurons}")
else:
    print(f"Using all {N_IN} neurons")

# ============================================================
# Helper functions
# ============================================================

def spikes_to_temporal_sequence(spike_i, spike_t, n_neurons, selected_neurons, bin_width_ms=50):
    """Convert spike trains to temporal sequence, optionally filtering neurons."""
    if len(spike_i) == 0:
        if selected_neurons:
            return np.zeros((len(selected_neurons), 1), dtype=np.float32)
        return np.zeros((n_neurons, 1), dtype=np.float32)

    max_t = np.max(spike_t) + 1e-6
    n_bins = int(np.ceil(max_t / bin_width_ms))
    n_bins = max(n_bins, 1)

    if selected_neurons:
        # Use only selected neurons
        neuron_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_neurons)}
        counts = np.zeros((len(selected_neurons), n_bins), dtype=np.float32)
        for ni, ti in zip(spike_i, spike_t):
            if ni in neuron_map:
                new_idx = neuron_map[ni]
                bin_idx = int(np.floor(ti / bin_width_ms))
                bin_idx = min(bin_idx, n_bins - 1)
                counts[new_idx, bin_idx] += 1
        return counts
    else:
        # Use all neurons
        counts = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for ni, ti in zip(spike_i, spike_t):
            bin_idx = int(np.floor(ti / bin_width_ms))
            bin_idx = min(bin_idx, n_bins - 1)
            counts[ni, bin_idx] += 1
        return counts


# ============================================================
# Process samples (SINGLE LOOP: sequences + raster data)
# ============================================================

all_sequences = []
all_spike_data = []
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

    # Extract spike times
    spike_i = np.array(mon.i, dtype=np.int32)
    spike_t = np.array(mon.t / ms, dtype=np.float32)

    # Convert to sequence with optional neuron filtering
    spike_sequence = spikes_to_temporal_sequence(spike_i, spike_t, N_IN, neurons, bin_width_ms=50)

    all_sequences.append(spike_sequence)

    parts = rel.replace("\\", "/").split("/")
    sample_name = parts[-1]
    sample_names.append(sample_name)

    # Store spike data for raster (filter if neurons list is not empty)
    spike_i_raster = spike_i
    spike_t_raster = spike_t
    if neurons:
        mask = np.isin(spike_i_raster, neurons)
        spike_i_raster = spike_i_raster[mask]
        spike_t_raster = spike_t_raster[mask]
    all_spike_data.append((spike_i_raster, spike_t_raster, duration_ms))

    print(f"  spikes={len(spike_i):,}  sequence_shape={spike_sequence.shape}")

    # ====================== Save sequence as heatmap ======================
    print(f"  Saving sequence heatmap...")

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(spike_sequence,
                   aspect='auto',
                   cmap='viridis',
                   interpolation='nearest',
                   origin='lower',
                   vmin=0)

    plt.colorbar(im, ax=ax, label='Spike Count per 50ms Bin')

    n_neurons_used = len(neurons) if neurons else N_IN
    ax.set_xlabel('Time Bin (each bin = 50 ms)', fontsize=12)
    ax.set_ylabel(f'Neuron Index ({"selected" if neurons else "0 to 699"})', fontsize=12)
    ax.set_title(f'Spike Count Heatmap - {sample_name}\n'
                 f'Shape: {spike_sequence.shape[0]} neurons × {spike_sequence.shape[1]} bins | '
                 f'Total spikes: {len(spike_i):,}',
                 fontsize=13)

    # Better y-ticks
    step = max(10, n_neurons_used // 10)
    ax.set_yticks(np.arange(0, n_neurons_used, step))

    plt.tight_layout()

    heatmap_filename = f"sequence_heatmap_{sample_idx:02d}_{os.path.splitext(sample_name)[0]}.png"
    heatmap_path = os.path.join(OUTPUT_DIR, heatmap_filename)
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {heatmap_path}")
    plt.close()

n = len(all_sequences)
print(f"\nProcessed {n} samples")

# ============================================================
# Compute Spikernel similarity matrix
# ============================================================

print(f"\n{'='*60}")
print(f"Computing {n}×{n} spikernel similarity matrix")
print(f"Hyperparameters: n_max={SPIKERNEL_N_MAX}, lam={SPIKERNEL_LAM}, mu={SPIKERNEL_MU}, q={SPIKERNEL_Q}")
print(f"{'='*60}\n")

K = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for j in range(n):
        K[i, j] = spikernel_normalized(
            all_sequences[i],
            all_sequences[j],
            n_max=SPIKERNEL_N_MAX,
            lam=SPIKERNEL_LAM,
            mu=SPIKERNEL_MU,
            q=SPIKERNEL_Q,
            eps=SPIKERNEL_EPS
        )
    if (i + 1) % max(1, n // 5) == 0:
        print(f"  Processed {i + 1}/{n} samples")

print(f"Similarity range: [{K.min():.4f}, {K.max():.4f}]")

# ============================================================
# Plot raster plots
# ============================================================

print("\nPlotting raster plots...")

# Plot rasters
for sample_idx, (spike_i, spike_t, duration_ms) in enumerate(all_spike_data):
    fig, ax = plt.subplots(figsize=(12, 8))

    if len(spike_i) > 0:
        ax.scatter(spike_t, spike_i, s=1.5, alpha=0.7, color="black")

    for t in np.arange(0, duration_ms + 1000, 1000):
        ax.axvline(x=t, color="red", linestyle="-", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Neuron Index", fontsize=12)
    title_str = f"Spike Raster - Sample {sample_idx} ({sample_names[sample_idx]})"
    if neurons:
        title_str += f"\n(Only {len(neurons)} selected neurons)"
    ax.set_title(title_str, fontsize=12)

    ax.set_xlim(0, duration_ms)
    if neurons:
        ax.set_ylim(min(neurons)-5, max(neurons)+5)
    else:
        ax.set_ylim(-1, N_IN)

    plt.tight_layout()

    raster_path = os.path.join(OUTPUT_DIR, f"raster_sample_{sample_idx:02d}.png")
    plt.savefig(raster_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {raster_path}")
    plt.close()

# ============================================================
# Plot similarity heatmap
# ============================================================

print("\nPlotting similarity heatmap...")

n = len(sample_names)
fig_size = max(8, n * 0.9 + 2)
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

ax.set_title(f"Input Layer Spike Similarity (Spikernel)\n"
             f"{'Selected neurons only' if neurons else 'All neurons'}", fontsize=14)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "similarity.png")
plt.savefig(heatmap_path, dpi=180, bbox_inches="tight")
print(f"Saved: {heatmap_path}")
plt.close()

print(f"\n{'='*60}\nDone! Files saved in:\n"
      f"   • sequence_heatmap_XX_*.png\n"
      f"   • raster_sample_XX.png\n"
      f"   • similarity.png\n"
      f"{'='*60}")