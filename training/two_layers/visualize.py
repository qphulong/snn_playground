"""
visualize.py
------------
Load history.npz (produced by train.py) and save all enabled plots as PNGs
into training/two_layers/ next to this script.

Usage:
    python visualize.py                                          # auto-locate history.npz + record_config.yaml
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

# ── load config ───────────────────────────────────────────────────────────────

visualize_samples = None
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    visualize_samples = cfg.get("visualize_samples", None)

# ── load data ─────────────────────────────────────────────────────────────────

data = np.load(npz_path, allow_pickle=True)
keys = set(data.files)
print(f"Loaded : {npz_path}")
print(f"Keys   : {sorted(keys)}\n")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, ".")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output : {OUT_DIR}\n")

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {name}")


def _sample_indices(n_total):
    """Return the list of sample indices to visualize."""
    if visualize_samples and len(visualize_samples) > 0:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _window_mask(t, t_start, t_end):
    """Boolean mask for t within [t_start, t_end]. -1 means no bound."""
    mask = np.ones(len(t), dtype=bool)
    if t_start >= 0:
        mask &= t >= t_start
    if t_end >= 0:
        mask &= t <= t_end
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 1. Weight evolution — one PNG per synapse pair
# ══════════════════════════════════════════════════════════════════════════════

if "we_pairs" in keys and "we_values" in keys and "we_times_ms" in keys:
    pairs    = data["we_pairs"]      # (n_pairs, 2)
    values   = data["we_values"]     # (n_pairs, n_snaps)
    times_ms = data["we_times_ms"]   # (n_snaps,)

    for k, (pi, pj) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(times_ms, values[k], lw=1.5, color=f"C{k % 10}")
        ax.set_title(f"Weight Evolution  —  synapse in[{pi}] → hid[{pj}]",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, f"weight_evolution_in{pi:04d}_hid{pj:04d}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Membrane potential — input layer
# ══════════════════════════════════════════════════════════════════════════════

if "vmon_in_v_all" in keys and "vmon_in_t_all" in keys:
    neurons    = data["vmon_in_indices"]    # (n_neurons,)
    v_all      = data["vmon_in_v_all"]      # object array
    t_all      = data["vmon_in_t_all"]
    windows    = data["vmon_in_windows"]    # (n_neurons, 3): [nid, t_start, t_end]
    n_total    = int(data["vmon_in_n_samples"])
    v_th_in    = float(data["v_th_in"]) if "v_th_in" in keys else 1.0

    # Build window lookup by neuron id
    win_lookup = {int(row[0]): (float(row[1]), float(row[2])) for row in windows}

    for s in _sample_indices(n_total):
        v = v_all[s]   # (n_neurons, T)
        t = t_all[s]   # (T,)

        for k, nid in enumerate(neurons):
            t_start, t_end = win_lookup.get(int(nid), (-1.0, -1.0))
            mask = _window_mask(t, t_start, t_end)
            t_w  = t[mask]
            v_w  = v[k][mask]

            # Spike times: where v just crossed threshold (reset means v drops to 0 next step)
            # Detect from the raw trace: find local maxima >= threshold
            spike_times = t_w[np.where((v_w[:-1] < v_th_in) & (v_w[1:] >= v_th_in))[0] + 1] \
                if len(v_w) > 1 else np.array([])

            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(t_w, v_w, lw=0.8, color="steelblue", zorder=2, label="v")
            ax.axhline(v_th_in, color="crimson", lw=1.0, ls="--", zorder=3, label=f"threshold ({v_th_in})")

            # Spike markers
            if len(spike_times) > 0:
                ax.vlines(spike_times, ymin=v_th_in, ymax=v_th_in * 1.25,
                          color="crimson", lw=1.0, zorder=4, label=f"spikes ({len(spike_times)})")

            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            ax.set_title(f"Membrane Potential — Input Neuron {nid}  |  Sample {s}{win_str}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("v (a.u.)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            # y-axis: 0 at bottom
            y_lo = min(0.0, v_w.min() * 1.05) if len(v_w) else 0.0
            y_hi = max(v_th_in * 1.35, v_w.max() * 1.1) if len(v_w) else v_th_in * 1.5
            ax.set_ylim(y_lo, y_hi)
            plt.tight_layout()
            save(fig, f"vmon_input_sample{s:03d}_neuron{nid:04d}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Membrane potential — hidden layer
# ══════════════════════════════════════════════════════════════════════════════

if "vmon_hid_v_all" in keys and "vmon_hid_t_all" in keys:
    neurons  = data["vmon_hid_indices"]
    v_all    = data["vmon_hid_v_all"]
    vth_all  = data["vmon_hid_vth_all"]
    t_all    = data["vmon_hid_t_all"]
    windows  = data["vmon_hid_windows"]
    n_total  = int(data["vmon_hid_n_samples"])

    win_lookup = {int(row[0]): (float(row[1]), float(row[2])) for row in windows}

    for s in _sample_indices(n_total):
        v   = v_all[s]    # (n_neurons, T)
        vth = vth_all[s]  # (n_neurons, T)
        t   = t_all[s]    # (T,)

        for k, nid in enumerate(neurons):
            t_start, t_end = win_lookup.get(int(nid), (-1.0, -1.0))
            mask  = _window_mask(t, t_start, t_end)
            t_w   = t[mask]
            v_w   = v[k][mask]
            vth_w = vth[k][mask]

            # Spike detection: v crosses vth from below
            crossed = (v_w[:-1] < vth_w[:-1]) & (v_w[1:] >= vth_w[1:])
            spike_times = t_w[np.where(crossed)[0] + 1] if len(v_w) > 1 else np.array([])

            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(t_w, v_w,   lw=0.8,  color="steelblue", zorder=2, label="v")
            ax.plot(t_w, vth_w, lw=1.0,  color="crimson",   zorder=3,
                    ls="--", label="vth (adaptive)")

            if len(spike_times) > 0:
                # Draw spike markers at threshold level
                spike_vth = np.interp(spike_times, t_w, vth_w)
                ax.vlines(spike_times, ymin=spike_vth, ymax=spike_vth * 1.25,
                          color="crimson", lw=1.0, zorder=4, label=f"spikes ({len(spike_times)})")

            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            ax.set_title(f"Membrane Potential — Hidden Neuron {nid}  |  Sample {s}{win_str}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("v (a.u.)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            y_lo = min(0.0, v_w.min() * 1.05) if len(v_w) else 0.0
            y_hi = max(vth_w.max() * 1.35, v_w.max() * 1.1) if len(v_w) else 2.0
            ax.set_ylim(y_lo, y_hi)
            plt.tight_layout()
            save(fig, f"vmon_hidden_sample{s:03d}_neuron{nid:04d}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Spike raster — input layer
# ══════════════════════════════════════════════════════════════════════════════

if "raster_in_i" in keys and "raster_in_t" in keys:
    raster_i = data["raster_in_i"]          # object array
    raster_t = data["raster_in_t"]
    n_total  = int(data["raster_in_n_samples"])
    n_neurons = int(data["raster_in_n_neurons"])

    for s in _sample_indices(n_total):
        sp_i = raster_i[s]   # spike neuron indices
        sp_t = raster_t[s]   # spike times (ms)

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            ax.scatter(sp_t, sp_i, s=0.5, c="steelblue", linewidths=0, rasterized=True)
        ax.set_title(f"Spike Raster — Input Layer  |  Sample {s}  "
                     f"({len(sp_t):,} spikes)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        # x: 0 at left, y: 0 at bottom, higher index upward
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=n_neurons - 1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save(fig, f"raster_input_sample{s:03d}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Spike raster — hidden layer
# ══════════════════════════════════════════════════════════════════════════════

if "raster_hid_i" in keys and "raster_hid_t" in keys:
    raster_i  = data["raster_hid_i"]
    raster_t  = data["raster_hid_t"]
    n_total   = int(data["raster_hid_n_samples"])
    n_neurons = int(data["raster_hid_n_neurons"])

    for s in _sample_indices(n_total):
        sp_i = raster_i[s]
        sp_t = raster_t[s]

        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            ax.scatter(sp_t, sp_i, s=0.5, c="mediumseagreen", linewidths=0, rasterized=True)
        ax.set_title(f"Spike Raster — Hidden Layer  |  Sample {s}  "
                     f"({len(sp_t):,} spikes)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=n_neurons - 1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save(fig, f"raster_hidden_sample{s:03d}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Final weight matrix
# ══════════════════════════════════════════════════════════════════════════════

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
        origin="lower",          # row 0 at bottom, higher index upward
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


# ══════════════════════════════════════════════════════════════════════════════
# 7. Mean firing rate — input layer (per neuron)
# ══════════════════════════════════════════════════════════════════════════════

if "mean_firing_rate_input" in keys:
    rates   = data["mean_firing_rate_input"]   # (N_IN,)
    n_in    = len(rates)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n_in), rates, width=1.0, color="darkorange", linewidth=0)
    ax.set_title("Mean Firing Rate — Input Layer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n_in - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "mean_firing_rate_input.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Mean firing rate — hidden layer (per neuron)
# ══════════════════════════════════════════════════════════════════════════════

if "mean_firing_rate_hidden" in keys:
    rates = data["mean_firing_rate_hidden"]   # (N_H,)
    n_h   = len(rates)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n_h), rates, width=1.0, color="mediumseagreen", linewidth=0)
    ax.set_title("Mean Firing Rate — Hidden Layer", fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n_h - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "mean_firing_rate_hidden.png")


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
if not saved:
    print("No recognised keys found — nothing was plotted.")