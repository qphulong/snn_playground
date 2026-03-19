"""
visualize.py
------------
Load history.npz (produced by train.py) and save all enabled plots as PNGs
into training/two_layers/ next to this script.

Usage:
    python visualize.py                        # looks for history.npz next to this script
    python visualize.py path/to/history.npz
    python visualize.py path/to/history.npz path/to/record_config.yaml
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

# ── resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

npz_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "history.npz")
cfg_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "record_config.yaml")

if not os.path.exists(npz_path):
    print(f"ERROR: history file not found: {npz_path}")
    sys.exit(1)

# ── load config (for visualize_samples) ──────────────────────────────────────

visualize_samples = None   # None = all
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    visualize_samples = cfg.get("visualize_samples", None)  # list[int] or null

# ── load data ─────────────────────────────────────────────────────────────────

data = np.load(npz_path, allow_pickle=True)
keys = set(data.files)
print(f"Loaded: {npz_path}")
print(f"Keys present: {sorted(keys)}\n")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, ".")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Saving PNGs to: {OUT_DIR}\n")

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {name}")


# ── 1. Weight evolution ───────────────────────────────────────────────────────

if "we_pairs" in keys and "we_values" in keys and "we_times_ms" in keys:
    pairs    = data["we_pairs"]      # (n_pairs, 2)
    values   = data["we_values"]     # (n_pairs, n_snaps)
    times_ms = data["we_times_ms"]   # (n_snaps,)

    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        n_pairs, 1,
        figsize=(11, 3 * n_pairs),
        sharex=True,
        squeeze=False
    )
    fig.suptitle("Weight Evolution", fontsize=14, fontweight="bold")

    for k, (pi, pj) in enumerate(pairs):
        ax = axes[k][0]
        ax.plot(times_ms, values[k], lw=1.5, color=f"C{k}")
        ax.set_ylabel(f"w[{pi},{pj}]", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)

    axes[-1][0].set_xlabel("Time (ms)")
    plt.tight_layout()
    save(fig, "weight_evolution.png")


# ── 2. Membrane potential — one PNG per neuron per sample ─────────────────────

if "vmon_v_all" in keys and "vmon_t_all" in keys and "vmon_neurons" in keys:
    neurons  = data["vmon_neurons"]          # (n_neurons,)
    v_all    = data["vmon_v_all"]            # object array of (n_neurons, T) per sample
    t_all    = data["vmon_t_all"]            # object array of (T,) per sample
    n_samples_total = int(data["vmon_n_samples"])

    # Determine which samples to visualize
    if visualize_samples and len(visualize_samples) > 0:
        sample_indices = [s for s in visualize_samples if s < n_samples_total]
    else:
        sample_indices = list(range(n_samples_total))

    for s in sample_indices:
        v = v_all[s]   # (n_neurons, T)
        t = t_all[s]   # (T,)

        for k, nid in enumerate(neurons):
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(t, v[k], lw=0.8, color="steelblue")
            ax.set_title(
                f"Membrane Potential — Hidden Neuron {nid}  |  Sample {s}",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("v (a.u.)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, f"membrane_potential_sample{s:03d}_neuron{nid:04d}.png")


# ── 3. Final weight matrix ────────────────────────────────────────────────────

if "final_weights_matrix" in keys:
    W = data["final_weights_matrix"]   # (N_IN, N_H)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    fig.suptitle("Final Weight Matrix", fontsize=14, fontweight="bold")

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        W,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=W.max() if W.max() > 0 else 1,
    )
    ax_heat.set_xlabel("Hidden neuron index")
    ax_heat.set_ylabel("Input neuron index")
    plt.colorbar(im, ax=ax_heat, label="Weight")

    ax_hist = fig.add_subplot(gs[1])
    w_flat = W.flatten()
    ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue", edgecolor="none", density=True)
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution\n(non-zero)")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, "final_weight_matrix.png")


# ── 4. Mean firing rate — input layer (per neuron) ───────────────────────────

if "mean_firing_rate_input" in keys:
    rates   = data["mean_firing_rate_input"]   # (N_IN,)
    n_in    = len(rates)
    indices = np.arange(n_in)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(indices, rates, width=1.0, color="darkorange", linewidth=0)
    ax.set_title("Mean Firing Rate — Input Layer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n_in - 0.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "mean_firing_rate_input.png")


# ── 5. Mean firing rate — hidden layer (per neuron) ──────────────────────────

if "mean_firing_rate_hidden" in keys:
    rates  = data["mean_firing_rate_hidden"]   # (N_H,)
    n_h    = len(rates)
    indices = np.arange(n_h)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(indices, rates, width=1.0, color="mediumseagreen", linewidth=0)
    ax.set_title("Mean Firing Rate — Hidden Layer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n_h - 0.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "mean_firing_rate_hidden.png")


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
if not saved:
    print("No recognised keys found — nothing was plotted.")
    print("Expected keys: we_pairs/we_values/we_times_ms, vmon_v_all/vmon_t_all/vmon_neurons,")
    print("  final_weights_matrix, mean_firing_rate_input, mean_firing_rate_hidden")