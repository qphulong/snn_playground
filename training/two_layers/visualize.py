#!/usr/bin/env python3
"""
Visualization script for two_layer.py training history.

Usage:
    python training/two_layers/visualize.py [options]

Examples:
    # Basic (all defaults)
    python training/two_layers/visualize.py

    # Custom neuron ranges and weight pairs
    python training/two_layers/visualize.py \\
        --in-range 0 200 \\
        --h-range 0 200 \\
        --weights 0,1 0,2 1,3 \\
        --output training/two_layers/plots/
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ============================================================
# Helpers
# ============================================================

def load_history(path):
    data = np.load(path, allow_pickle=False)
    return data


def compute_spike_rates(spike_i, spike_t, n_neurons, T_ms, bin_ms=50.0):
    """Return per-spike firing rate (Hz) based on the time bin each spike falls in."""
    n_bins = max(1, int(np.ceil(T_ms / bin_ms)))
    bin_idx = np.clip((spike_t / bin_ms).astype(np.int32), 0, n_bins - 1)

    counts = np.zeros((n_neurons, n_bins), dtype=np.float32)
    np.add.at(counts, (spike_i, bin_idx), 1)
    rates = counts / (bin_ms / 1000.0)  # Hz

    return rates[spike_i, bin_idx]


# ============================================================
# Plot 1 – Spike raster (input + hidden)
# ============================================================

def plot_raster(data, in_range, h_range, output_dir):
    in_i = data["in_spike_i"]
    in_t = data["in_spike_t"]
    h_i  = data["h_spike_i"]
    h_t  = data["h_spike_t"]
    T_ms = float(data["T_ms"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Spike Raster Plot", fontsize=14)

    for ax, spike_i, spike_t, rng, title in [
        (axes[0], in_i, in_t, in_range, "Input Layer"),
        (axes[1], h_i,  h_t,  h_range,  "Hidden Layer"),
    ]:
        lo, hi = rng
        mask = (spike_i >= lo) & (spike_i < hi)
        ax.scatter(spike_t[mask], spike_i[mask], s=0.3, c="black", rasterized=True)
        ax.set_xlim(0, T_ms)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Neuron index")
        ax.set_title(f"{title}  (neurons {lo}–{hi-1})")

    axes[1].set_xlabel("Time (ms)")
    plt.tight_layout()

    out = os.path.join(output_dir, "raster.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Plot 2 – Weight matrix heatmap
# ============================================================

def plot_weight_heatmap(data, output_dir):
    w = data["w_final"]   # (N_IN, N_H)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(w, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_xlabel("Hidden neuron index")
    ax.set_ylabel("Input neuron index")
    ax.set_title("Final Weight Matrix Heatmap")
    plt.tight_layout()

    out = os.path.join(output_dir, "weight_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Plot 3 – Rate-colored spike raster (input + hidden)
# ============================================================

def plot_rate_raster(data, in_range, h_range, bin_ms, output_dir):
    in_i = data["in_spike_i"]
    in_t = data["in_spike_t"]
    h_i  = data["h_spike_i"]
    h_t  = data["h_spike_t"]
    T_ms = float(data["T_ms"])
    N_IN = int(data["N_IN"])
    N_H  = int(data["N_H"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Rate-Colored Raster  (bin = {bin_ms} ms)", fontsize=14)

    for ax, spike_i, spike_t, n_neurons, rng, title in [
        (axes[0], in_i, in_t, N_IN, in_range, "Input Layer"),
        (axes[1], h_i,  h_t,  N_H,  h_range,  "Hidden Layer"),
    ]:
        lo, hi = rng
        mask = (spike_i >= lo) & (spike_i < hi)
        t_sel = spike_t[mask]
        i_sel = spike_i[mask]

        if len(t_sel) == 0:
            ax.set_title(f"{title} — no spikes in range")
            continue

        rates = compute_spike_rates(i_sel, t_sel, n_neurons, T_ms, bin_ms)
        cmap = cm.hot_r
        norm = Normalize(vmin=0, vmax=rates.max() if rates.max() > 0 else 1)

        ax.scatter(t_sel, i_sel, s=0.3, c=rates, cmap=cmap, norm=norm,
                   rasterized=True)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Firing rate (Hz)")

        ax.set_xlim(0, T_ms)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Neuron index")
        ax.set_title(f"{title}  (neurons {lo}–{hi-1})")

    axes[1].set_xlabel("Time (ms)")
    plt.tight_layout()

    out = os.path.join(output_dir, "rate_raster.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Plot 4 – Weight evolution for selected (pre, post) pairs
# ============================================================

def plot_weight_evolution(data, weight_pairs, output_dir):
    w_snaps = data["w_snapshots"]          # (n_snaps, N_IN, N_H)
    times   = data["w_snapshot_times_ms"]  # (n_snaps,)

    if w_snaps.ndim != 3 or w_snaps.shape[0] == 0:
        print("No weight snapshots available — skipping weight evolution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for pre, post in weight_pairs:
        values = w_snaps[:, pre, post]
        ax.plot(times, values, marker="o", markersize=3, label=f"({pre}→{post})")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.set_title("Weight Evolution of Selected Synapses")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, "weight_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Main
# ============================================================

def parse_weight_pairs(raw):
    """Parse list of 'i,j' strings into list of (int, int) tuples."""
    pairs = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Weight pair must be 'i,j', got: {s!r}"
            )
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_history = os.path.join(script_dir, "history.npz")

    parser = argparse.ArgumentParser(
        description="Visualize two_layer.py training history."
    )
    parser.add_argument(
        "--history", default=default_history,
        help="Path to history.npz (default: same directory as this script)",
    )
    parser.add_argument(
        "--in-range", nargs=2, type=int, default=[0, 100], metavar=("LO", "HI"),
        help="Neuron index range [LO, HI) for input raster (default: 0 100)",
    )
    parser.add_argument(
        "--h-range", nargs=2, type=int, default=[0, 100], metavar=("LO", "HI"),
        help="Neuron index range [LO, HI) for hidden raster (default: 0 100)",
    )
    parser.add_argument(
        "--weights", nargs="+", default=["0,1", "0,2", "1,3"],
        metavar="I,J",
        help="Synapse pairs for weight evolution, e.g. 0,1 0,2 1,3",
    )
    parser.add_argument(
        "--bin-ms", type=float, default=50.0,
        help="Time bin size in ms for rate-colored raster (default: 50)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory to save plots (default: same directory as history.npz)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.history):
        print(f"Error: history file not found: {args.history}")
        return

    output_dir = args.output or os.path.dirname(os.path.abspath(args.history))
    os.makedirs(output_dir, exist_ok=True)

    weight_pairs = parse_weight_pairs(args.weights)

    print(f"Loading: {args.history}")
    data = load_history(args.history)

    in_range = tuple(args.in_range)
    h_range  = tuple(args.h_range)

    plot_raster(data, in_range, h_range, output_dir)
    plot_weight_heatmap(data, output_dir)
    plot_rate_raster(data, in_range, h_range, args.bin_ms, output_dir)
    plot_weight_evolution(data, weight_pairs, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
