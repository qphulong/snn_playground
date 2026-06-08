"""
visualize.py
------------
Load epoch history files (history_epoch_*.npz) and save all enabled plots as PNGs.

Works with the registry recorder key schema:
  group keys   : {group_name}__{field}        e.g. hidden__raster_i
  synapse keys : {synapse_name}__{field}      e.g. in_to_hid__we_pairs
  (-> in names is stored as _to_)

Output structure:
  vizs/epoch_init/    — initial weight matrices
  vizs/epoch_0/       — plots for epoch 0
  vizs/epoch_1/       — plots for epoch 1
  ...

Usage:
    python visualize.py
"""

import sys
import os
import glob as glob_mod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yaml

# ── resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path   = os.path.join(SCRIPT_DIR, "record_and_visualize_config.yaml")

# ── load config ───────────────────────────────────────────────────────────────

visualize_samples    = None
visualize_epoch      = []
weights_per_neuron   = {}   # {synapse_name: [neuron_ids]}
track_weight_delta   = False
group_cfg            = {}
synapse_cfg          = {}
input_neuron_layout  = {}   # per-channel input neuron type layout (for type-coloured plots)

if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Support both the new split format and the legacy flat format.
    viz_cfg = cfg.get("visualize", cfg)
    visualize_samples   = viz_cfg.get("visualize_samples",  None)
    visualize_epoch     = viz_cfg.get("visualize_epoch",    [])
    weights_per_neuron  = viz_cfg.get("weights_per_neuron", {}) or {}
    track_weight_delta  = viz_cfg.get("track_weight_delta", False)
    group_cfg           = viz_cfg.get("groups",   {}) or {}
    synapse_cfg         = viz_cfg.get("synapses", {}) or {}
    input_neuron_layout = viz_cfg.get("input_neuron_layout", {}) or {}

# ── find epoch files ──────────────────────────────────────────────────────────

epoch_files = sorted(glob_mod.glob(os.path.join(SCRIPT_DIR, "history_epoch_*.npz")))
if not epoch_files:
    print(f"ERROR: No history_epoch_*.npz files found in {SCRIPT_DIR}")
    sys.exit(1)

print(f"Found {len(epoch_files)} epoch file(s)")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, "vizs")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output base: {OUT_DIR}\n")

saved = []


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, name, epoch_dir=None):
    out = OUT_DIR if epoch_dir is None else os.path.join(OUT_DIR, epoch_dir)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {os.path.relpath(path, SCRIPT_DIR)}")


def _sample_indices(n_total):
    if visualize_samples:
        return [s for s in visualize_samples if s < n_total]
    return list(range(n_total))


def _epochs_to_visualize(total):
    if visualize_epoch:
        return [e for e in visualize_epoch if e < total]
    return list(range(total))


def _window_mask(t, t_start, t_end):
    mask = np.ones(len(t), dtype=bool)
    if t_start >= 0:
        mask &= t >= t_start
    if t_end >= 0:
        mask &= t <= t_end
    return mask


def _pfx(name):
    """Convert a group/synapse name to its npz key prefix."""
    return name.replace("->", "_to_")


def _k(name, field):
    """Full npz key for a group or synapse field."""
    return f"{_pfx(name)}__{field}"


def _has(keys, name, field):
    return _k(name, field) in keys


def _get(data, name, field):
    return data[_k(name, field)]


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers  (style copied from original visualize.py)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_weight_matrix(W, title, filename, epoch_dir=None):
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        W, aspect="auto", interpolation="nearest", cmap="viridis",
        vmin=0, vmax=W.max() if W.max() > 0 else 1, origin="lower",
    )
    ax_heat.set_xlabel("Post-synaptic neuron index")
    ax_heat.set_ylabel("Pre-synaptic neuron index")
    plt.colorbar(im, ax=ax_heat, label="Weight")

    ax_hist = fig.add_subplot(gs[1])
    w_flat = W.flatten()
    ax_hist.hist(w_flat[w_flat > 0], bins=60, color="steelblue", edgecolor="none", density=True)
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution\n(non-zero)")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, filename, epoch_dir=epoch_dir)


def _plot_firing_rate(rates, title, filename, color="darkorange", epoch_dir=None,
                      legend_handles=None):
    n = len(rates)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n), rates, width=1.0, color=color, linewidth=0)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, filename, epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Per-group plot routines
# ══════════════════════════════════════════════════════════════════════════════

def _plot_spike_raster(data, keys, name, epoch_idx, epoch_dir, color):
    if not (_has(keys, name, "raster_i") and _has(keys, name, "raster_t")):
        return
    raster_i  = _get(data, name, "raster_i")
    raster_t  = _get(data, name, "raster_t")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))

    type_colors, handles = _input_type_colors(name, n_neurons)

    for s in _sample_indices(n_total):
        sp_i = raster_i[s]
        sp_t = raster_t[s]
        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            c = (list(type_colors[sp_i.astype(np.int32)])
                 if type_colors is not None else color)
            ax.scatter(sp_t, sp_i, s=0.5, c=c, linewidths=0, rasterized=True)
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8, markerscale=2)
        ax.set_title(
            f"Spike Raster — {name}  |  Epoch {epoch_idx}, Sample {s}  ({len(sp_t):,} spikes)",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=n_neurons - 1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save(fig, f"raster_{_pfx(name)}_sample{s:03d}.png", epoch_dir=epoch_dir)


def _plot_spike_counts(data, keys, name, epoch_idx, epoch_dir, color):
    """Spike count bar chart per neuron (derived from raster)."""
    if not _has(keys, name, "raster_i"):
        return
    raster_i  = _get(data, name, "raster_i")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))

    type_colors, handles = _input_type_colors(name, n_neurons)
    bar_color = type_colors if type_colors is not None else color

    for s in _sample_indices(n_total):
        sp_i = raster_i[s]
        counts = (np.bincount(sp_i.astype(np.int32), minlength=n_neurons)
                  if len(sp_i) > 0 else np.zeros(n_neurons, dtype=np.int32))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(n_neurons), counts, width=1.0, color=bar_color, linewidth=0)
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_title(
            f"Spike Count — {name}  |  Epoch {epoch_idx}, Sample {s}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Spike count")
        ax.set_xlim(-0.5, n_neurons - 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        save(fig, f"spike_count_{_pfx(name)}_sample{s:03d}.png", epoch_dir=epoch_dir)


def _plot_mean_firing_rate(data, keys, name, epoch_idx, epoch_dir, color):
    pfx = _pfx(name)

    def _colors_for(n_neurons):
        type_colors, handles = _input_type_colors(name, n_neurons)
        return (type_colors if type_colors is not None else color), handles

    # Whole-dataset aggregate
    if _has(keys, name, "mfr"):
        mfr = _get(data, name, "mfr")
        bar_color, handles = _colors_for(len(mfr))
        _plot_firing_rate(
            mfr,
            f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, all samples",
            f"mean_firing_rate_{pfx}_all.png",
            color=bar_color, epoch_dir=epoch_dir, legend_handles=handles
        )

    # Per-sample
    if (_has(keys, name, "mfr_sample_counts") and
            _has(keys, name, "mfr_sample_dur_s")):
        sample_counts = _get(data, name, "mfr_sample_counts")
        sample_durs   = _get(data, name, "mfr_sample_dur_s")
        n_total       = len(sample_durs)

        for s in _sample_indices(n_total):
            dur   = float(sample_durs[s])
            rates = (sample_counts[s] / dur if dur > 0
                     else np.zeros_like(sample_counts[s], dtype=np.float32))
            bar_color, handles = _colors_for(len(rates))
            _plot_firing_rate(
                rates.astype(np.float32),
                f"Mean Firing Rate — {name}  |  Epoch {epoch_idx}, Sample {s}",
                f"mean_firing_rate_{pfx}_sample{s:03d}.png",
                color=bar_color, epoch_dir=epoch_dir, legend_handles=handles
            )


def _plot_membrane_potential(data, keys, name, epoch_idx, epoch_dir):
    v_key  = _k(name, "vmon_v_all")
    t_key  = _k(name, "vmon_t_all")
    if v_key not in keys or t_key not in keys:
        return

    neurons   = _get(data, name, "vmon_indices")
    t_all     = _get(data, name, "vmon_t_all")
    windows   = _get(data, name, "vmon_windows")
    n_total   = len(t_all)
    pfx       = _pfx(name)

    # Collect all variable arrays that were recorded for this group
    # Keys look like: {pfx}__vmon_{var}_all
    var_keys = {}
    for k in keys:
        tag = f"{_pfx(name)}__vmon_"
        if k.startswith(tag) and k.endswith("_all"):
            var = k[len(tag):-4]   # strip prefix and _all suffix
            if var != "t":         # t_all handled separately
                var_keys[var] = k

    if not var_keys:
        return

    for s in _sample_indices(n_total):
        t = t_all[s]

        for k_idx, nid in enumerate(neurons):
            t_start = float(windows[k_idx, 1]) if windows[k_idx, 1] >= 0 else -1.0
            t_end   = float(windows[k_idx, 2]) if windows[k_idx, 2] >= 0 else -1.0
            mask    = _window_mask(t, t_start, t_end)
            t_w     = t[mask]

            n_vars = len(var_keys)
            fig, axes = plt.subplots(n_vars, 1, figsize=(11, 3 * n_vars), sharex=True,
                                     squeeze=False)
            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            fig.suptitle(
                f"State Variables — {name}, Neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}{win_str}",
                fontsize=11, fontweight="bold"
            )

            colors = ["steelblue", "crimson", "darkorange", "mediumseagreen",
                      "mediumpurple", "saddlebrown"]

            for ax_i, (var, full_key) in enumerate(sorted(var_keys.items())):
                ax  = axes[ax_i, 0]
                arr = data[full_key][s]   # shape (n_monitored_neurons, n_timesteps)
                y_w = arr[k_idx][mask]
                ax.plot(t_w, y_w, lw=0.8, color=colors[ax_i % len(colors)], label=var)
                ax.set_ylabel(var)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)

            axes[-1, 0].set_xlabel("Time (ms)")
            plt.tight_layout()
            win_suffix = (f"_window{t_start:.0f}_{t_end:.0f}ms" if t_start >= 0 else "")
            save(fig,
                 f"vmon_{pfx}_sample{s:03d}_neuron{nid:04d}{win_suffix}.png",
                 epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Per-synapse plot routines
# ══════════════════════════════════════════════════════════════════════════════

def _plot_weight_evolution(data, keys, name, epoch_idx, epoch_dir):
    if not _has(keys, name, "we_pairs"):
        return
    pairs    = _get(data, name, "we_pairs")    # (n_pairs, 2)
    values   = _get(data, name, "we_values")   # (n_pairs, n_total_snaps)
    pfx      = _pfx(name)

    # Whole-dataset: x = snapshot index
    x_axis  = np.arange(values.shape[1])
    x_label = "Snapshot index"

    for k, (pi, pj) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, values[k], lw=1.5, color=f"C{k % 10}")
        ax.set_title(
            f"Weight Evolution (epoch {epoch_idx}, all samples) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, f"weight_evolution_{pfx}_all_pre{pi:04d}_post{pj:04d}.png",
             epoch_dir=epoch_dir)

    # Per-sample
    if (_has(keys, name, "we_sample_values") and
            _has(keys, name, "we_sample_times")):
        sample_values = _get(data, name, "we_sample_values")
        sample_times  = _get(data, name, "we_sample_times")
        n_total       = len(sample_times)

        for s in _sample_indices(n_total):
            sv = sample_values[s]   # (n_pairs, n_snaps)
            st = sample_times[s]    # (n_snaps,)
            if len(st) == 0:
                continue
            st_rel = st - st[0]

            for k, (pi, pj) in enumerate(pairs):
                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(st_rel, sv[k], lw=1.5, color=f"C{k % 10}")
                ax.set_title(
                    f"Weight Evolution (epoch {epoch_idx}, sample {s}) — "
                    f"{name}  pre[{pi}] → post[{pj}]",
                    fontsize=12, fontweight="bold"
                )
                ax.set_xlabel("Time within sample (ms)")
                ax.set_ylabel("Weight")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save(fig,
                     f"weight_evolution_{pfx}_sample{s:03d}_pre{pi:04d}_post{pj:04d}.png",
                     epoch_dir=epoch_dir)


def _plot_weight_delta(data, keys, name, epoch_idx, epoch_dir):
    if not track_weight_delta:
        return
    if not _has(keys, name, "we_pairs"):
        return
    pairs  = _get(data, name, "we_pairs")
    values = _get(data, name, "we_values")
    pfx    = _pfx(name)

    x_axis = np.arange(values.shape[1])

    for k, (pi, pj) in enumerate(pairs):
        w_vals = values[k]
        deltas = np.diff(w_vals, prepend=w_vals[0])
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, deltas, lw=1.5, color=f"C{k % 10}", marker="o", markersize=3)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(
            f"Weight Delta (epoch {epoch_idx}) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Δw")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, f"weight_delta_{pfx}_all_pre{pi:04d}_post{pj:04d}.png",
             epoch_dir=epoch_dir)


def _plot_synapse_weight_matrix(data, keys, name, epoch_idx, epoch_dir):
    pfx = _pfx(name)

    # Final weight matrix (end of epoch)
    if _has(keys, name, "final_weights"):
        W = _get(data, name, "final_weights")
        _plot_weight_matrix(
            W,
            f"Final Weight Matrix — {name}  |  Epoch {epoch_idx}",
            f"final_weight_matrix_{pfx}.png",
            epoch_dir=epoch_dir
        )

    # Per-sample snapshots
    if (_has(keys, name, "weight_per_sample") and
            _has(keys, name, "weight_n_samples")):
        wm_per_sample = _get(data, name, "weight_per_sample")
        n_total       = int(_get(data, name, "weight_n_samples"))

        for s in _sample_indices(n_total):
            W = wm_per_sample[s]
            _plot_weight_matrix(
                W,
                f"Weight Matrix — {name}  |  Epoch {epoch_idx}, after sample {s}",
                f"weight_matrix_{pfx}_sample{s:03d}.png",
                epoch_dir=epoch_dir
            )


def _plot_weights_per_neuron(data, keys, name, epoch_idx, epoch_dir):
    # weights_per_neuron is {synapse_name: [post-synaptic neuron ids]}
    neuron_ids = weights_per_neuron.get(name, [])
    if not neuron_ids:
        return
    if not (_has(keys, name, "weight_per_sample") and
            _has(keys, name, "weight_n_samples")):
        return
    wm_per_sample = _get(data, name, "weight_per_sample")
    n_total       = int(_get(data, name, "weight_n_samples"))

    for s in _sample_indices(n_total):
        W = wm_per_sample[s]   # (N_pre, N_post)

        for nid in neuron_ids:
            nid = int(nid)
            if nid >= W.shape[1]:
                continue
            weights = W[:, nid]   # incoming weights for this post-synaptic neuron

            fig = plt.figure(figsize=(14, 5))
            gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
            fig.suptitle(
                f"Incoming Weights — {name}, post-neuron {nid}  |  "
                f"Epoch {epoch_idx}, Sample {s}",
                fontsize=13, fontweight="bold"
            )

            ax_bar = fig.add_subplot(gs[0])
            ax_bar.bar(np.arange(len(weights)), weights, width=0.8,
                       color="steelblue", linewidth=0)
            ax_bar.set_xlabel("Input neuron index")
            ax_bar.set_ylabel("Weight magnitude")
            ax_bar.set_xlim(-0.5, len(weights) - 0.5)
            ax_bar.set_ylim(bottom=0)
            ax_bar.grid(True, axis="y", alpha=0.3)

            ax_hist = fig.add_subplot(gs[1])
            w_nonzero = weights[weights > 0]
            if len(w_nonzero) > 0:
                ax_hist.hist(w_nonzero, bins=40, color="steelblue",
                             edgecolor="none", density=True)
            ax_hist.set_xlabel("Weight value")
            ax_hist.set_ylabel("Density")
            ax_hist.set_title("Distribution\n(non-zero)")
            ax_hist.grid(True, alpha=0.3)

            plt.tight_layout()
            save(fig,
                 f"weights_per_neuron_{_pfx(name)}_sample{s:03d}_neuron{nid:04d}.png",
                 epoch_dir=epoch_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Colour palette per group (add more as needed)
# ══════════════════════════════════════════════════════════════════════════════

GROUP_COLORS = {
    "input":  "steelblue",
    "hidden": "mediumseagreen",
    "output": "darkorange",
}

def _group_color(name):
    return GROUP_COLORS.get(name, "mediumpurple")


# ── Colour input neurons by spike type (sustained / onset / phase) ────────────
# Input neurons are laid out per frequency channel as
#   [sustained × S, onset × O, phase × P]  (see compute_spike_input_current),
# so neuron i's type is determined by (i % per_band).  The layout is read from
# record_and_visualize_config.yaml -> visualize.input_neuron_layout and must
# match the encoding used in train.py.

TYPE_COLORS = {
    "sustained": "steelblue",
    "onset":     "crimson",
    "phase":     "darkorange",
}


def _input_type_colors(name, n_neurons):
    """Per-neuron colour array + legend handles for type-coloured input plots.

    Returns (colors, handles) where ``colors`` is an (n_neurons,) array of colour
    strings, or (None, None) if ``name`` is not the configured input group or no
    layout is available (callers then fall back to the single group colour).
    """
    if not input_neuron_layout:
        return None, None
    if name != input_neuron_layout.get("group", "input"):
        return None, None

    s = int(input_neuron_layout.get("sustained_per_band", 0))
    o = int(input_neuron_layout.get("onset_per_band",     0))
    p = int(input_neuron_layout.get("phase_per_band",     0))
    per_band = s + o + p
    if per_band <= 0:
        return None, None

    pos    = np.arange(n_neurons) % per_band
    colors = np.empty(n_neurons, dtype=object)
    colors[pos < s]                       = TYPE_COLORS["sustained"]
    colors[(pos >= s) & (pos < s + o)]    = TYPE_COLORS["onset"]
    colors[pos >= s + o]                  = TYPE_COLORS["phase"]

    handles = []
    if s > 0:
        handles.append(mpatches.Patch(color=TYPE_COLORS["sustained"], label=f"sustained ({s}/band)"))
    if o > 0:
        handles.append(mpatches.Patch(color=TYPE_COLORS["onset"], label=f"onset ({o}/band)"))
    if p > 0:
        handles.append(mpatches.Patch(color=TYPE_COLORS["phase"], label=f"phase ({p}/band)"))
    return colors, handles


# ══════════════════════════════════════════════════════════════════════════════
# Main epoch loop
# ══════════════════════════════════════════════════════════════════════════════

for epoch_idx, npz_path in enumerate(_epochs_to_visualize(len(epoch_files))):
    # _epochs_to_visualize returns indices, not paths — fix:
    pass

for epoch_idx, npz_path in enumerate(epoch_files):
    if epoch_idx not in _epochs_to_visualize(len(epoch_files)):
        continue

    epoch_dir = f"epoch_{epoch_idx}"
    print(f"\nProcessing Epoch {epoch_idx} — {os.path.basename(npz_path)}")

    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    # ── Groups ────────────────────────────────────────────────────────────────
    for name in group_cfg:
        color = _group_color(name)
        _plot_spike_raster(data, keys, name, epoch_idx, epoch_dir, color)
        _plot_spike_counts(data, keys, name, epoch_idx, epoch_dir, color)
        _plot_mean_firing_rate(data, keys, name, epoch_idx, epoch_dir, color)
        _plot_membrane_potential(data, keys, name, epoch_idx, epoch_dir)

    # ── Synapses ──────────────────────────────────────────────────────────────
    for name in synapse_cfg:
        _plot_weight_evolution(data, keys, name, epoch_idx, epoch_dir)
        _plot_weight_delta(data, keys, name, epoch_idx, epoch_dir)
        _plot_synapse_weight_matrix(data, keys, name, epoch_idx, epoch_dir)
        _plot_weights_per_neuron(data, keys, name, epoch_idx, epoch_dir)

    # ── Initial weight matrices (epoch 0 only, stored inside epoch npz) ──────
    if epoch_idx == 0:
        for name in synapse_cfg:
            if _has(keys, name, "init_weights"):
                W = _get(data, name, "init_weights")
                _plot_weight_matrix(
                    W,
                    f"Initial Weight Matrix — {name}",
                    f"init_weight_matrix_{_pfx(name)}.png",
                    epoch_dir="epoch_init"
                )


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved to: {OUT_DIR}")
if not saved:
    print("No recognised keys found in npz files — nothing was plotted.")
    print("Check that group/synapse names in record_config.yaml match those "
          "passed to recorder.track_group() / recorder.track_synapses().")