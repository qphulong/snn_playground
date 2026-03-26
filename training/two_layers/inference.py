"""
inference.py
============
Runs the trained SNN on inference audio files and computes pairwise
similarity between all samples using the spikernel.

Audio paths (recursive scan):
    datasets/vox1_small_infer_test/**/*.wav

Spike sequences:
    - Neurons  = hidden layer (G_h), N_H = 700
    - Time bin = 100 ms  →  spike counts per neuron per bin

Similarity:
    K(s_i, s_j) via spikernel(), plotted as a heatmap with file-path labels.
"""

import os, sys, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from brian2 import *
from src.utils.spike_encoding import compute_spike_input_current
from src.spikernel import spikernel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "history_epoch_0.npz")   # change epoch if needed
INFER_DIR      = "datasets/vox1_tiny/test"
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vizs")

BIN_MS         = 100       # spike-count bin width (ms)
N_MAX          = 5         # spikernel N_MAX
LAM            = 1.0       # spikernel lambda
GAMMA          = 1.0       # spikernel gamma

# ─────────────────────────────────────────────────────────────────────────────
# Network hyperparameters  (must match training)
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 700
N_H  = 700
DT_SIM = 1 * ms

# Input layer
tau_m       = 50 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 0.15
v_th_in     = 1.0

# Hidden layer
tau_h    = 50 * ms
tau_vth  = 200 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.2

# Synaptic weight bounds
wmax = 0.1
wmin = 0.0

# Lateral inhibition
lat_inh = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load trained weights
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading weights from: {WEIGHTS_PATH}")
data = np.load(WEIGHTS_PATH, allow_pickle=True)
w_matrix = data["final_weights_matrix"].astype(np.float64)   # (N_IN, N_H)
assert w_matrix.shape == (N_IN, N_H), f"Expected ({N_IN},{N_H}), got {w_matrix.shape}"
print(f"  weight matrix shape : {w_matrix.shape}")
print(f"  weight range        : [{w_matrix.min():.4f}, {w_matrix.max():.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Collect audio files
# ─────────────────────────────────────────────────────────────────────────────

wav_files = sorted(glob.glob(os.path.join(INFER_DIR, "**", "*.wav"), recursive=True))
print(f"\nFound {len(wav_files)} wav files under {INFER_DIR}")
for f in wav_files:
    print(f"  {f}")

if len(wav_files) == 0:
    raise RuntimeError(f"No .wav files found under {INFER_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run network on each file → collect hidden-layer spike sequences
# ─────────────────────────────────────────────────────────────────────────────

def run_network(audio_path, w_matrix):
    """
    Run the trained SNN on one audio file.
    Returns spike_i (array of neuron indices) and spike_t_ms (array of spike
    times in ms) for the hidden layer.
    """
    I, T = compute_spike_input_current(
        audio_path,
        scale=1,
        sustained_per_band=4,
        onset_per_band=2,
        phase_per_band=1,
        sust_spread_min=0.7,
        sust_spread_max=1.3,
    )

    start_scope()
    defaultclock.dt = DT_SIM

    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    # Input layer
    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t,i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(
        N_IN, eqs_in,
        threshold="v > v_th_in",
        reset="v=0; a+=beta",
        refractory=2*ms,
        method="euler",
    )

    # Hidden layer
    eqs_h = f"""
    dv/dt  = -v / tau_h                      : 1
    dvth/dt = -(vth - {vth_rest}) / tau_vth  : 1
    """
    G_h = NeuronGroup(
        N_H, eqs_h,
        threshold="v > vth",
        reset=f"v=0; vth=vth+{vth_jump};",
        refractory=2*ms,
        method="euler",
    )
    G_h.vth = vth_init

    # Synapses (fixed weights — no STDP at inference)
    S = Synapses(G_in, G_h, "w : 1", on_pre="v_post += w")
    S.connect()
    src = np.array(S.i)
    tgt = np.array(S.j)
    S.w = w_matrix[src, tgt]

    # Lateral inhibition
    lat = Synapses(G_h, G_h, on_pre="v_post = clip(v_post * 0.8, 0, inf)")
    lat.connect(condition="i != j")

    # Monitor hidden layer only
    hid_spike_mon = SpikeMonitor(G_h)

    run(T * DT_SIM)

    spike_i  = np.array(hid_spike_mon.i,      dtype=np.int32)
    spike_t  = np.array(hid_spike_mon.t / ms, dtype=np.float32)
    duration_ms = float(T) * float(DT_SIM / ms)

    return spike_i, spike_t, duration_ms


def spikes_to_binned_sequence(spike_i, spike_t_ms, n_neurons, duration_ms, bin_ms):
    """
    Convert spike trains into a 2-D spike-count matrix.

    Returns:
        seq : ndarray, shape (n_neurons, n_bins)  — integer spike counts
    """
    n_bins = max(1, int(np.ceil(duration_ms / bin_ms)))
    seq = np.zeros((n_neurons, n_bins), dtype=np.float32)
    if len(spike_i) == 0:
        return seq
    bin_idx = np.floor(spike_t_ms / bin_ms).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for ni, bi in zip(spike_i, bin_idx):
        seq[ni, bi] += 1
    return seq


print(f"\n{'='*60}")
print("Running inference on all files …")
print(f"{'='*60}")

sequences = []      # list of (n_neurons, n_bins) arrays
labels    = []      # short display label  (folder/filename)
start_all = time.time()

for idx, path in enumerate(wav_files):
    rel = os.path.relpath(path, start=os.path.dirname(INFER_DIR))
    print(f"[{idx+1}/{len(wav_files)}] {rel}")
    t0 = time.time()

    try:
        spike_i, spike_t, dur_ms = run_network(path, w_matrix)
    except Exception as e:
        print(f"  ERROR: {e}  — skipping")
        continue

    seq = spikes_to_binned_sequence(spike_i, spike_t, N_H, dur_ms, BIN_MS)
    sequences.append(seq)

    # Label: last two path components (speaker / file)
    parts = rel.replace("\\", "/").split("/")
    label = "/".join(parts[-2:]) if len(parts) >= 2 else rel
    labels.append(label)

    n_spikes = int(spike_i.shape[0])
    print(f"  spikes={n_spikes:,}  duration={dur_ms:.0f} ms  "
          f"bins={seq.shape[1]}  elapsed={time.time()-t0:.1f}s")

n = len(sequences)
print(f"\nSuccessfully processed {n}/{len(wav_files)} files  "
      f"(total {time.time()-start_all:.1f}s)")

if n == 0:
    raise RuntimeError("No sequences were produced — check audio paths / model.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute pairwise spikernel similarity matrix
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nComputing {n}×{n} spikernel similarity matrix …")
print(f"  N_MAX={N_MAX}  LAM={LAM}  GAMMA={GAMMA}")

K = np.zeros((n, n), dtype=np.float64)
t_k = time.time()
for i in range(n):
    for j in range(i, n):
        val = spikernel(sequences[i], sequences[j], N_MAX, LAM, GAMMA)
        K[i, j] = val
        K[j, i] = val
    print(f"  row {i+1}/{n} done")

print(f"  kernel matrix computed in {time.time()-t_k:.1f}s")

# Normalise to cosine-style similarity in [0, 1]
diag = np.sqrt(np.diag(K))
diag_safe = np.where(diag == 0, 1.0, diag)
K_norm = K / np.outer(diag_safe, diag_safe)
K_norm = np.clip(K_norm, 0.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot heatmap
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

    # Annotate cells with value
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            txt_color = "white" if val < 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(5, 9 - n // 4), color=txt_color)

    # Draw grid lines between cells
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Highlight diagonal
    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="red",
                                   linewidth=1.5, zorder=3))

    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("Audio file")
    ax.set_ylabel("Audio file")

    # ── Audio-path trail annotation ───────────────────────────────────────────
    # Show the folder (speaker) for each label as a colour-coded strip on top
    speakers = [lbl.split("/")[0] for lbl in labels]
    unique_sp = list(dict.fromkeys(speakers))          # ordered-unique
    cmap_sp   = plt.cm.get_cmap("tab10", len(unique_sp))
    sp_colors = {sp: cmap_sp(i) for i, sp in enumerate(unique_sp)}

    strip_h = 0.018   # fraction of figure height
    for k, sp in enumerate(speakers):
        color = sp_colors[sp]
        # top strip (x-axis)
        ax.add_patch(plt.Rectangle(
            (k - 0.5, -1.5), 1, 0.5,
            color=color, clip_on=False, transform=ax.transData, zorder=4))
        # left strip (y-axis)
        ax.add_patch(plt.Rectangle(
            (-1.5, k - 0.5), 0.5, 1,
            color=color, clip_on=False, transform=ax.transData, zorder=4))

    # Legend for speaker colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=sp_colors[sp], label=sp) for sp in unique_sp]
    ax.legend(handles=legend_handles, title="Speaker / folder",
              bbox_to_anchor=(1.18, 1), loc="upper left",
              fontsize=8, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {save_path}")


heatmap_path = os.path.join(OUTPUT_DIR, "spikernel_similarity.png")
make_heatmap(
    K_norm, labels,
    title=(f"Spikernel similarity  (N_MAX={N_MAX}, λ={LAM}, γ={GAMMA}, bin={BIN_MS}ms)"),
    save_path=heatmap_path,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save raw kernel matrix
# ─────────────────────────────────────────────────────────────────────────────

npz_path = os.path.join(OUTPUT_DIR, "spikernel_matrix.npz")
np.savez_compressed(
    npz_path,
    K_raw=K.astype(np.float32),
    K_norm=K_norm.astype(np.float32),
    labels=np.array(labels),
)
print(f"Raw kernel matrix saved → {npz_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Console summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Inference complete!")
print(f"  Files processed : {n}")
print(f"  Kernel range    : [{K.min():.4f}, {K.max():.4f}]")
print(f"  Norm sim range  : [{K_norm.min():.4f}, {K_norm.max():.4f}]")
print(f"  Heatmap         : {heatmap_path}")
print(f"  Kernel matrix   : {npz_path}")
print(f"{'='*60}")