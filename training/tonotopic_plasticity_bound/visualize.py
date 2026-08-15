"""
visualize.py
------------
Load per-piece history files. Each piece_XXXX/ directory owns its own
history_epoch_YYY.npz files (one per repeat/epoch, its own 0..N-1 numbering)
and save all enabled plots as PNGs.

Works with the registry recorder key schema:
  group keys   : {group_name}__{field}        e.g. hidden__raster_i
  synapse keys : {synapse_name}__{field}      e.g. in_to_hid__we_pairs
  (-> in names is stored as _to_)

Output structure:
  vizs/piece_0000/epoch_init/  — piece 0000's initial weight matrices
  vizs/piece_0000/epoch_0/     — piece 0000's plots for epoch 0
  vizs/piece_0000/epoch_1/     — piece 0000's plots for epoch 1
  ...
  vizs/piece_0001/epoch_0/     — piece 0001's plots for epoch 0
  ...
  vizs/fingerprints/           — one converged weight-matrix image per piece

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
visualize_pieces     = []
weights_per_neuron   = {}   # {synapse_name: [neuron_ids]}
track_weight_delta   = False
group_cfg            = {}
synapse_cfg          = {}

if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Support both the new split format and the legacy flat format.
    viz_cfg = cfg.get("visualize", cfg)
    visualize_samples  = viz_cfg.get("visualize_samples",  None)
    visualize_epoch    = viz_cfg.get("visualize_epoch",    [])
    visualize_pieces   = viz_cfg.get("visualize_pieces",   [])
    weights_per_neuron = viz_cfg.get("weights_per_neuron", {}) or {}
    track_weight_delta = viz_cfg.get("track_weight_delta", False)
    group_cfg          = viz_cfg.get("groups",   {}) or {}
    synapse_cfg        = viz_cfg.get("synapses", {}) or {}

# ── find piece directories — each piece owns its own epoch_000..NNN.npz files ──

piece_dirs = sorted(
    d for d in glob_mod.glob(os.path.join(SCRIPT_DIR, "piece_*"))
    if os.path.isdir(d)
)
if not piece_dirs:
    print(f"ERROR: No piece_* directories found in {SCRIPT_DIR}")
    sys.exit(1)

print(f"Found {len(piece_dirs)} piece director{'y' if len(piece_dirs) == 1 else 'ies'}")

# ── output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(SCRIPT_DIR, "vizs")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output base: {OUT_DIR}\n")

saved   = []
skipped = []


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _out_path(name, epoch_dir=None):
    out = OUT_DIR if epoch_dir is None else os.path.join(OUT_DIR, epoch_dir)
    return os.path.join(out, name)


def _already_saved(name, epoch_dir=None):
    """True if this PNG was produced by a previous run. Callers check this
    BEFORE building the figure (not just before writing it) so a re-run can
    skip the expensive matplotlib rendering entirely, not just the file I/O."""
    return os.path.exists(_out_path(name, epoch_dir))


def _skip(name, epoch_dir=None):
    skipped.append(_out_path(name, epoch_dir))


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


def _pieces_to_visualize(total):
    if visualize_pieces:
        return [p for p in visualize_pieces if p < total]
    return list(range(total))


_current_piece_label = ""   # set once per piece directory in the main loop


def _ptitle(epoch_idx):
    """Combined 'piece_label, Epoch N' label used in every plot title."""
    return f"{_current_piece_label}, Epoch {epoch_idx}"


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
    if _already_saved(filename, epoch_dir):
        _skip(filename, epoch_dir)
        return
    w_min, w_max = W.min(), W.max()
    if w_max <= w_min:
        w_max = w_min + 1e-8

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(
        W, aspect="auto", interpolation="nearest", cmap="viridis",
        vmin=w_min, vmax=w_max, origin="lower",
    )
    ax_heat.set_xlabel("Post-synaptic neuron index")
    ax_heat.set_ylabel("Pre-synaptic neuron index")
    plt.colorbar(im, ax=ax_heat, label="Weight")

    ax_hist = fig.add_subplot(gs[1])
    w_flat = W.flatten()
    nz = w_flat[w_flat > 0]
    if len(nz):
        ax_hist.hist(nz, bins=60, range=(w_min, w_max), color="steelblue", edgecolor="none", density=True)
    ax_hist.set_xlim(w_min, w_max)
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution\n(non-zero)")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    save(fig, filename, epoch_dir=epoch_dir)


def _plot_firing_rate(rates, title, filename, color="darkorange", epoch_dir=None, handles=None):
    if _already_saved(filename, epoch_dir):
        _skip(filename, epoch_dir)
        return
    n = len(rates)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(n), rates, width=1.0, color=color, linewidth=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    if handles is not None:
        ax.legend(handles=handles, loc="upper right", fontsize=9)
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
    color_array, handles = _neuron_type_colors(name, n_neurons)

    for s in _sample_indices(n_total):
        fname = f"raster_{_pfx(name)}_rep{s:03d}.png"
        if _already_saved(fname, epoch_dir):
            _skip(fname, epoch_dir)
            continue
        sp_i = raster_i[s]
        sp_t = raster_t[s]
        fig, ax = plt.subplots(figsize=(12, 5))
        if len(sp_t) > 0:
            c = color_array[sp_i.astype(int)] if color_array is not None else color
            ax.scatter(sp_t, sp_i, s=3, c=c, linewidths=0, rasterized=True)
        ax.set_title(
            f"Spike Raster — {name}  |  {_ptitle(epoch_idx)}  ({len(sp_t):,} spikes)",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=n_neurons - 1)
        ax.grid(True, alpha=0.2)
        if handles is not None:
            ax.legend(handles=handles, loc="upper right", fontsize=9, markerscale=2)
        plt.tight_layout()
        save(fig, fname, epoch_dir=epoch_dir)


def _plot_spike_counts(data, keys, name, epoch_idx, epoch_dir, color):
    """Spike count bar chart per neuron (derived from raster)."""
    if not _has(keys, name, "raster_i"):
        return
    raster_i  = _get(data, name, "raster_i")
    n_total   = int(_get(data, name, "raster_n_samples"))
    n_neurons = int(_get(data, name, "raster_n_neurons"))
    color_array, handles = _neuron_type_colors(name, n_neurons)
    bar_color = color_array if color_array is not None else color

    for s in _sample_indices(n_total):
        fname = f"spike_count_{_pfx(name)}_rep{s:03d}.png"
        if _already_saved(fname, epoch_dir):
            _skip(fname, epoch_dir)
            continue
        sp_i = raster_i[s]
        counts = (np.bincount(sp_i.astype(np.int32), minlength=n_neurons)
                  if len(sp_i) > 0 else np.zeros(n_neurons, dtype=np.int32))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(n_neurons), counts, width=1.0, color=bar_color, linewidth=0)
        ax.set_title(
            f"Spike Count — {name}  |  {_ptitle(epoch_idx)}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Spike count")
        ax.set_xlim(-0.5, n_neurons - 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        if handles is not None:
            ax.legend(handles=handles, loc="upper right", fontsize=9)
        plt.tight_layout()
        save(fig, fname, epoch_dir=epoch_dir)


def _plot_mean_firing_rate(data, keys, name, epoch_idx, epoch_dir, color):
    pfx = _pfx(name)

    # Whole-dataset aggregate
    if _has(keys, name, "mfr"):
        rates = _get(data, name, "mfr")
        carr, handles = _neuron_type_colors(name, len(rates))
        _plot_firing_rate(
            rates,
            f"Mean Firing Rate — {name}  |  {_ptitle(epoch_idx)}",
            f"mean_firing_rate_{pfx}_all.png",
            color=carr if carr is not None else color,
            epoch_dir=epoch_dir, handles=handles
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
            carr, handles = _neuron_type_colors(name, len(rates))
            _plot_firing_rate(
                rates.astype(np.float32),
                f"Mean Firing Rate — {name}  |  {_ptitle(epoch_idx)}",
                f"mean_firing_rate_{pfx}_rep{s:03d}.png",
                color=carr if carr is not None else color,
                epoch_dir=epoch_dir, handles=handles
            )


def _plot_piece_membrane_potential(piece_path, piece_label):
    """Plot each piece's ONE continuous membrane-potential trace (spans all of
    that piece's repeats), read from piece_path/membrane_potential.npz — written
    once per piece by recorder.save_piece_membrane() for groups configured with
    `membrane_potential_scope: piece`. No-op if that file isn't present (e.g. for
    groups still using the default per-epoch scope, or older recordings)."""
    npz_path = os.path.join(piece_path, "membrane_potential.npz")
    if not os.path.exists(npz_path):
        return

    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    for name in group_cfg:
        t_key = _k(name, "vmon_t")
        if t_key not in keys:
            continue

        pfx     = _pfx(name)
        t       = _get(data, name, "vmon_t")
        neurons = _get(data, name, "vmon_indices")
        windows = _get(data, name, "vmon_windows")

        # Collect all variable arrays recorded for this group: {pfx}__vmon_{var}
        # (excluding the non-variable bookkeeping keys t/indices/windows)
        tag = f"{pfx}__vmon_"
        _non_var = {"t", "indices", "windows"}
        var_keys = {
            k[len(tag):]: k for k in keys
            if k.startswith(tag) and k[len(tag):] not in _non_var
        }
        if not var_keys:
            continue

        vmon_epoch_dir = os.path.join(piece_label, "membrane_potential")

        for k_idx, nid in enumerate(neurons):
            t_start = float(windows[k_idx, 1]) if windows[k_idx, 1] >= 0 else -1.0
            t_end   = float(windows[k_idx, 2]) if windows[k_idx, 2] >= 0 else -1.0
            win_suffix = (f"_window{t_start:.0f}_{t_end:.0f}ms" if t_start >= 0 else "")
            fname = f"vmon_{pfx}_fullpiece_neuron{nid:04d}{win_suffix}.png"
            if _already_saved(fname, vmon_epoch_dir):
                _skip(fname, vmon_epoch_dir)
                continue

            mask = _window_mask(t, t_start, t_end)
            t_w  = t[mask]

            n_vars = len(var_keys)
            fig, axes = plt.subplots(n_vars, 1, figsize=(11, 3 * n_vars), sharex=True,
                                     squeeze=False)
            win_str = f"  [{t_start:.0f}–{t_end:.0f} ms]" if t_start >= 0 else ""
            fig.suptitle(
                f"State Variables — {name}, Neuron {nid}  |  "
                f"{piece_label} (full piece){win_str}",
                fontsize=11, fontweight="bold"
            )

            colors = ["steelblue", "crimson", "darkorange", "mediumseagreen",
                      "mediumpurple", "saddlebrown"]

            for ax_i, (var, full_key) in enumerate(sorted(var_keys.items())):
                ax  = axes[ax_i, 0]
                arr = data[full_key]   # shape (n_monitored_neurons, n_timesteps)
                y_w = arr[k_idx][mask]
                ax.plot(t_w, y_w, lw=0.8, color=colors[ax_i % len(colors)], label=var)
                ax.set_ylabel(var)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)

            axes[-1, 0].set_xlabel("Time (ms)")
            plt.tight_layout()
            save(fig, fname, epoch_dir=vmon_epoch_dir)


def _plot_global_raster():
    """Plot the whole-clip, input-only, no-learning raster produced by train.py's
    pre-piece probe pass (SAVE_DIR/global_input_raster.npz). No-op if absent."""
    fname, epoch_dir = "raster_input_full_clip.png", "global"
    if _already_saved(fname, epoch_dir):
        _skip(fname, epoch_dir)
        return

    npz_path = os.path.join(SCRIPT_DIR, "global_input_raster.npz")
    if not os.path.exists(npz_path):
        return

    data      = np.load(npz_path, allow_pickle=True)
    raster_i  = data["raster_i"]
    raster_t  = data["raster_t"]
    n_neurons = int(data["n_neurons"])
    color_array, handles = _neuron_type_colors("input", n_neurons)

    duration_ms = float(data["duration_ms"]) if "duration_ms" in data.files else (
        float(raster_t.max()) if len(raster_t) > 0 else 0.0
    )

    fig, ax = plt.subplots(figsize=(20, 5))
    for t_mark in np.arange(0, duration_ms, 100):
        ax.axvline(t_mark, color="gray", lw=0.3, alpha=0.4, zorder=0)
    if len(raster_t) > 0:
        c = color_array[raster_i.astype(int)] if color_array is not None else "steelblue"
        ax.scatter(raster_t, raster_i, s=3, c=c, linewidths=0, rasterized=True, zorder=1)
    ax.set_title(
        f"Global Input Raster — full clip, no learning  ({len(raster_t):,} spikes)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=n_neurons - 1)
    ax.grid(True, alpha=0.2)
    if handles is not None:
        ax.legend(handles=handles, loc="upper right", fontsize=9, markerscale=2)
    plt.tight_layout()
    save(fig, fname, epoch_dir=epoch_dir)


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
        fname = f"weight_evolution_{pfx}_all_pre{pi:04d}_post{pj:04d}.png"
        if _already_saved(fname, epoch_dir):
            _skip(fname, epoch_dir)
            continue
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, values[k], lw=1.5, color=f"C{k % 10}")
        ax.set_title(
            f"Weight Evolution ({_ptitle(epoch_idx)}) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, fname, epoch_dir=epoch_dir)

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
                fname = f"weight_evolution_{pfx}_rep{s:03d}_pre{pi:04d}_post{pj:04d}.png"
                if _already_saved(fname, epoch_dir):
                    _skip(fname, epoch_dir)
                    continue
                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(st_rel, sv[k], lw=1.5, color=f"C{k % 10}")
                ax.set_title(
                    f"Weight Evolution ({_ptitle(epoch_idx)}) — "
                    f"{name}  pre[{pi}] → post[{pj}]",
                    fontsize=12, fontweight="bold"
                )
                ax.set_xlabel("Time within repeat (ms)")
                ax.set_ylabel("Weight")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                save(fig, fname, epoch_dir=epoch_dir)


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
        fname = f"weight_delta_{pfx}_all_pre{pi:04d}_post{pj:04d}.png"
        if _already_saved(fname, epoch_dir):
            _skip(fname, epoch_dir)
            continue
        w_vals = values[k]
        deltas = np.diff(w_vals, prepend=w_vals[0])
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(x_axis, deltas, lw=1.5, color=f"C{k % 10}", marker="o", markersize=3)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(
            f"Weight Delta ({_ptitle(epoch_idx)}) — "
            f"{name}  pre[{pi}] → post[{pj}]",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Δw")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, fname, epoch_dir=epoch_dir)


def _plot_synapse_weight_matrix(data, keys, name, epoch_idx, epoch_dir):
    pfx = _pfx(name)

    # Final weight matrix (end of piece)
    if _has(keys, name, "final_weights"):
        W = _get(data, name, "final_weights")
        _plot_weight_matrix(
            W,
            f"Final Weight Matrix — {name}  |  {_ptitle(epoch_idx)}",
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
                f"Weight Matrix — {name}  |  {_ptitle(epoch_idx)}",
                f"weight_matrix_{pfx}_rep{s:03d}.png",
                epoch_dir=epoch_dir
            )


def _export_fingerprint(piece_label, data, keys, epoch_idx):
    """Save ONE combined fingerprint image per piece — the in->hid weight matrix
    plus the input and hidden mean-firing-rate vectors, side by side — into a
    dedicated, flat `fingerprints/` directory (one file per piece, nothing else)
    so the sequence can be flipped through as a fingerprint movie. Built from
    the piece's LAST epoch (i.e. after all repeats), for EVERY piece regardless
    of `visualize_pieces` (the fingerprint movie always covers the whole clip)."""
    fname, epoch_dir = f"fingerprint_{piece_label}.png", "fingerprints"
    if _already_saved(fname, epoch_dir):
        _skip(fname, epoch_dir)
        return

    have_w   = _has(keys, "in->hid", "final_weights")
    have_in  = _has(keys, "input",  "mfr")
    have_hid = _has(keys, "hidden", "mfr")
    if not (have_w or have_in or have_hid):
        return

    def _rate_panel(ax, group_name, title):
        rates = _get(data, group_name, "mfr")
        carr, handles = _neuron_type_colors(group_name, len(rates))
        ax.bar(np.arange(len(rates)), rates, width=1.0,
               color=carr if carr is not None else _group_color(group_name), linewidth=0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Mean rate (Hz)")
        ax.set_xlim(-0.5, len(rates) - 0.5)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        if handles is not None:
            ax.legend(handles=handles, loc="upper right", fontsize=7)

    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], figure=fig)
    fig.suptitle(
        f"Fingerprint — {piece_label}  (after epoch {epoch_idx})",
        fontsize=14, fontweight="bold"
    )

    ax_w = fig.add_subplot(gs[0])
    if have_w:
        W = _get(data, "in->hid", "final_weights")
        w_min, w_max = W.min(), W.max()
        if w_max <= w_min:
            w_max = w_min + 1e-8
        im = ax_w.imshow(W, aspect="auto", interpolation="nearest", cmap="viridis",
                          vmin=w_min, vmax=w_max, origin="lower")
        ax_w.set_title("in->hid weights", fontsize=11, fontweight="bold")
        ax_w.set_xlabel("Post-synaptic neuron")
        ax_w.set_ylabel("Pre-synaptic neuron")
        plt.colorbar(im, ax=ax_w, label="Weight")
    else:
        ax_w.axis("off")

    ax_in = fig.add_subplot(gs[1])
    if have_in:
        _rate_panel(ax_in, "input", "Input rate")
    else:
        ax_in.axis("off")

    ax_hid = fig.add_subplot(gs[2])
    if have_hid:
        _rate_panel(ax_hid, "hidden", "Hidden rate")
    else:
        ax_hid.axis("off")

    plt.tight_layout()
    save(fig, fname, epoch_dir=epoch_dir)


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
            fname = f"weights_per_neuron_{_pfx(name)}_rep{s:03d}_neuron{nid:04d}.png"
            if _already_saved(fname, epoch_dir):
                _skip(fname, epoch_dir)
                continue
            weights = W[:, nid]   # incoming weights for this post-synaptic neuron

            fig = plt.figure(figsize=(14, 5))
            gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
            fig.suptitle(
                f"Incoming Weights — {name}, post-neuron {nid}  |  "
                f"{_ptitle(epoch_idx)}",
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
            save(fig, fname, epoch_dir=epoch_dir)


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


def _neuron_type_colors(name, n_neurons):
    """Per-neuron color array + legend handles from the group's `neuron_types` config.

    The neuron layout repeats `[count_0, count_1, ...]` per frequency band, matching
    compute_spike_input_current's (sustained, onset) ordering, so neuron i's
    type is decided by (i % neurons_per_band).

    Returns (None, None) when the group has no `neuron_types` config — callers then
    fall back to the single group color, leaving other groups unaffected.
    """
    types = group_cfg.get(name, {}).get("neuron_types")
    if not types:
        return None, None

    band = [(t["name"], int(t["count"]), t.get("color", "gray")) for t in types]
    per_band = sum(c for _, c, _ in band)
    if per_band <= 0:
        return None, None

    pos_color = []
    for _, cnt, col in band:
        pos_color.extend([col] * cnt)

    color_array = np.array([pos_color[i % per_band] for i in range(n_neurons)])
    handles = [mpatches.Patch(color=col, label=nm) for nm, _, col in band]
    return color_array, handles


# ══════════════════════════════════════════════════════════════════════════════
# Global (not piece-specific) plots
# ══════════════════════════════════════════════════════════════════════════════

_plot_global_raster()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop — pieces, then each piece's own epochs (repeats)
# ══════════════════════════════════════════════════════════════════════════════

for piece_num, piece_path in enumerate(piece_dirs):
    _current_piece_label = os.path.basename(piece_path)   # e.g. "piece_0000"

    piece_epoch_files = sorted(glob_mod.glob(os.path.join(piece_path, "history_epoch_*.npz")))
    if not piece_epoch_files:
        print(f"\nWARNING: no history_epoch_*.npz found in {piece_path}, skipping")
        continue

    # The fingerprint movie always covers every piece; the detailed per-epoch
    # plot suite below still honors visualize_pieces.
    do_full_viz = piece_num in _pieces_to_visualize(len(piece_dirs))

    for epoch_idx, npz_path in enumerate(piece_epoch_files):
        is_final_epoch = (epoch_idx == len(piece_epoch_files) - 1)

        if is_final_epoch:
            fp_data = np.load(npz_path, allow_pickle=True)
            fp_keys = set(fp_data.files)
            _export_fingerprint(_current_piece_label, fp_data, fp_keys, epoch_idx)

        if not do_full_viz or epoch_idx not in _epochs_to_visualize(len(piece_epoch_files)):
            continue

        epoch_dir = os.path.join(_current_piece_label, f"epoch_{epoch_idx}")
        print(f"\nProcessing {_ptitle(epoch_idx)} — {os.path.basename(npz_path)}")

        data = fp_data if is_final_epoch else np.load(npz_path, allow_pickle=True)
        keys = fp_keys if is_final_epoch else set(data.files)

        # ── Groups ────────────────────────────────────────────────────────────
        for name in group_cfg:
            color = _group_color(name)
            _plot_spike_raster(data, keys, name, epoch_idx, epoch_dir, color)
            _plot_spike_counts(data, keys, name, epoch_idx, epoch_dir, color)
            _plot_mean_firing_rate(data, keys, name, epoch_idx, epoch_dir, color)

        # ── Synapses ──────────────────────────────────────────────────────────
        for name in synapse_cfg:
            _plot_weight_evolution(data, keys, name, epoch_idx, epoch_dir)
            _plot_weight_delta(data, keys, name, epoch_idx, epoch_dir)
            _plot_synapse_weight_matrix(data, keys, name, epoch_idx, epoch_dir)
            _plot_weights_per_neuron(data, keys, name, epoch_idx, epoch_dir)

        # ── Initial weight matrices (piece's first epoch only) ──────────────────
        if epoch_idx == 0:
            for name in synapse_cfg:
                if _has(keys, name, "init_weights"):
                    W = _get(data, name, "init_weights")
                    _plot_weight_matrix(
                        W,
                        f"Initial Weight Matrix — {name}  |  {_current_piece_label}",
                        f"init_weight_matrix_{_pfx(name)}.png",
                        epoch_dir=os.path.join(_current_piece_label, "epoch_init")
                    )

    # ── Membrane potential: one continuous whole-piece trace, not per-epoch ────
    if do_full_viz:
        _plot_piece_membrane_potential(piece_path, _current_piece_label)


# ── finish ────────────────────────────────────────────────────────────────────

print(f"\n{len(saved)} PNG(s) saved, {len(skipped)} already existed (skipped) — output: {OUT_DIR}")
if not saved and not skipped:
    print("No recognised keys found in npz files — nothing was plotted.")
    print("Check that group/synapse names in record_config.yaml match those "
          "passed to recorder.track_group() / recorder.track_synapses().")