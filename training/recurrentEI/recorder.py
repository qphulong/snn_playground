"""
recorder.py
-----------
Recording module for SNN training.  Reads record_config.yaml,
attaches Brian2 monitors to a built network, accumulates data
across samples, and saves per-epoch .npz files.

Public API
----------
    rec_cfg  = load_record_config(path)
    recorder = Recorder(rec_cfg, arch)

    # ── inside sample loop ──
    ops = recorder.setup_sample(net)          # attach monitors, return ops list
    # ... brian2 run(T * dt) ...
    recorder.collect_sample(net, ops, T_steps)

    # ── after sample loop ──
    recorder.save_epoch(epoch_idx, out_dir, w_current)

    # ── after all epochs ──
    recorder.log_top_k(w_final, out_dir)

Design
------
- All group/synapse references in record_config.yaml use the same
  names as architecture.yaml.
- Spike monitors for groups listed in spike_raster OR mean_firing_rate
  are created automatically.  Groups needed for normalisation (always
  the destination groups of trainable synapses) are also monitored
  and their SpikeMonitor exposed via ops.norm_spike_monitors.
- The Recorder is stateless between epochs; call reset_epoch() or
  re-instantiate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
from brian2 import (
    SpikeMonitor,
    StateMonitor,
    NetworkOperation,
    defaultclock,
    ms,
)

from network_builder import NetworkObjects, extract_weights, _parse_synapse_name


# ── helpers ───────────────────────────────────────────────────────────────────

def load_record_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _pack_ragged(lst: list) -> np.ndarray:
    """Pack a list of variable-length arrays into a 1-D object array."""
    out = np.empty(len(lst), dtype=object)
    for k, a in enumerate(lst):
        out[k] = np.asarray(a)
    return out


def _parse_vmon_entries(raw: list) -> tuple[list[int], list[Any]]:
    """Parse membrane potential neuron entries.
    Returns (indices, windows) where windows[k] is None or (t0, t1).
    """
    indices, windows = [], []
    for entry in raw or []:
        if isinstance(entry, (int, float)):
            indices.append(int(entry))
            windows.append(None)
        else:
            indices.append(int(entry[0]))
            win = tuple(entry[1]) if len(entry) > 1 and entry[1] is not None else None
            windows.append(win)
    return indices, windows


def _make_flat_idx(src: np.ndarray, tgt: np.ndarray, pairs: list) -> list:
    """Pre-compute flat synapse indices for a list of (i,j) pairs."""
    flat = []
    for pi, pj in pairs:
        mask = (src == pi) & (tgt == pj)
        idx  = np.where(mask)[0]
        flat.append(int(idx[0]) if len(idx) > 0 else None)
    return flat


# ── per-sample operational state ──────────────────────────────────────────────

@dataclass
class SampleOps:
    """Holds all Brian2 monitors and network_operations for one sample."""
    # spike monitors keyed by group name
    spike_monitors: dict[str, SpikeMonitor]          = field(default_factory=dict)
    # membrane monitors keyed by group name
    vmon:           dict[str, StateMonitor]          = field(default_factory=dict)
    # weight snapshot accumulators keyed by synapse name
    snap_times:     dict[str, list]                  = field(default_factory=dict)
    snap_vals:      dict[str, list]                  = field(default_factory=dict)  # [k] = list of floats
    snap_flat_idx:  dict[str, list]                  = field(default_factory=dict)
    # network_operation objects (must be in Brian2 run scope)
    net_ops:        list                             = field(default_factory=list)


# ── weight snapshot callable ─────────────────────────────────────────────────
# Brian2 requires network_operation functions to have exactly one argument (t)
# or no arguments.  A callable class sidesteps this restriction cleanly.

class _WeightSnapper:
    """Callable that Brian2 accepts as a NetworkOperation.
    Brian2 calls network operation functions with zero arguments at runtime
    (operations.py: self.function()), so __call__ must take no arguments.
    We read the current clock time via defaultclock.t instead.
    """

    def __init__(self, syn, flat_idx, snap_times, snap_vals, ep_times, ep_vals,
                 clock_ref):
        self._syn        = syn
        self._flat_idx   = flat_idx
        self._snap_times = snap_times
        self._snap_vals  = snap_vals
        self._ep_times   = ep_times
        self._ep_vals    = ep_vals
        self._clock      = clock_ref   # brian2 Clock object

    def __call__(self):
        t_ms  = float(self._clock.t / ms)
        w_arr = np.array(self._syn.w)
        self._snap_times.append(t_ms)
        self._ep_times.append(t_ms)
        for k, fidx in enumerate(self._flat_idx):
            val = float(w_arr[fidx]) if fidx is not None else float("nan")
            self._snap_vals[k].append(val)
            self._ep_vals[k].append(val)


# ── main recorder ─────────────────────────────────────────────────────────────

class Recorder:
    """Manages monitor attachment, data accumulation, and npz saving."""

    def __init__(self, rec_cfg: dict, arch: dict):
        self.cfg  = rec_cfg
        self.arch = arch
        self._init_epoch_store()

    # ── epoch state ───────────────────────────────────────────────────────────

    def _init_epoch_store(self):
        """Initialise per-epoch accumulators."""
        cfg  = self.cfg
        arch = self.arch

        self._raster_groups   = set(cfg.get("spike_raster",      []) or [])
        self._mfr_groups      = set(cfg.get("mean_firing_rate",  []) or [])
        self._need_spike      = self._raster_groups | self._mfr_groups

        # Groups that are dst of trainable synapses → always need spike monitor
        for sname, scfg in arch.get("synapse_groups", {}).items():
            if scfg.get("plasticity", "none") in ("stdp", "static"):
                _, dst = _parse_synapse_name(sname)
                self._need_spike.add(dst)

        # Membrane potential: group_name → (indices, windows)
        self._vmon_cfg: dict[str, tuple[list, list]] = {}
        for gname, entries in (cfg.get("membrane_potential") or {}).items():
            idx, win = _parse_vmon_entries(entries)
            if idx:
                self._vmon_cfg[gname] = (idx, win)

        # Weight evolution: synapse_name → list of [src_id, dst_id]
        self._we_cfg: dict[str, list] = {}
        for sname, pairs in (cfg.get("weights_evolution") or {}).items():
            if pairs:
                self._we_cfg[sname] = [list(p) for p in pairs]

        self._final_w_syns   = set(cfg.get("final_weights", []) or [])
        self._snap_interval  = float(cfg.get("snapshot_interval_ms", 500))

        # ── accumulators (reset each epoch) ──────────────────────────────────
        self.ep: dict[str, Any] = {}

        # spikes
        for gname in (self._raster_groups | self._mfr_groups):
            self.ep[f"spikes_{gname}_i"] = []
            self.ep[f"spikes_{gname}_t"] = []

        # vmon
        for gname, (idx, win) in self._vmon_cfg.items():
            self.ep[f"vmon_{gname}_v"] = []
            self.ep[f"vmon_{gname}_t"] = []
            self.ep[f"vmon_{gname}_indices"] = idx
            self.ep[f"vmon_{gname}_windows"] = win

        # weight evolution
        for sname, pairs in self._we_cfg.items():
            self.ep[f"we_{sname}_pairs"]        = pairs
            self.ep[f"we_{sname}_times_ms"]     = []
            self.ep[f"we_{sname}_values"]       = [[] for _ in pairs]
            self.ep[f"we_{sname}_sample_times"] = []
            self.ep[f"we_{sname}_sample_vals"]  = []

        # final weights
        for sname in self._final_w_syns:
            self.ep[f"w_{sname}_per_sample"] = []

        self.ep["sample_durations_s"] = []
        self.ep["sample_sim_durations_s"] = []

    def reset_epoch(self):
        self._init_epoch_store()

    # ── per-sample: attach monitors ───────────────────────────────────────────

    def setup_sample(
        self,
        net: NetworkObjects,
        dt_ms: float = 1.0,
    ) -> SampleOps:
        """
        Create and attach all Brian2 monitors for one sample.
        Returns a SampleOps that must be passed to collect_sample().
        The caller must include ops.net_ops in the Brian2 run() scope
        (or add them to a Network object).
        """
        ops = SampleOps()
        snap_ms = self._snap_interval

        # ── Spike monitors ────────────────────────────────────────────────────
        for gname in self._need_spike:
            if gname in net.groups:
                ops.spike_monitors[gname] = SpikeMonitor(net.groups[gname])

        # ── Membrane monitors ─────────────────────────────────────────────────
        for gname, (idx, _) in self._vmon_cfg.items():
            if gname in net.groups:
                ops.vmon[gname] = StateMonitor(net.groups[gname], "v", record=idx)

        # ── Weight snapshot network_operations ───────────────────────────────
        for sname, pairs in self._we_cfg.items():
            if sname not in net.synapses:
                continue

            flat_idx = _make_flat_idx(
                net.src_idx[sname], net.tgt_idx[sname], pairs
            )
            n_pairs = len(pairs)

            ops.snap_times[sname]    = []
            ops.snap_vals[sname]     = [[] for _ in range(n_pairs)]
            ops.snap_flat_idx[sname] = flat_idx

            snapper = _WeightSnapper(
                syn        = net.synapses[sname],
                flat_idx   = flat_idx,
                snap_times = ops.snap_times[sname],
                snap_vals  = ops.snap_vals[sname],
                ep_times   = self.ep[f"we_{sname}_times_ms"],
                ep_vals    = self.ep[f"we_{sname}_values"],
                clock_ref  = defaultclock,
            )
            net_op = NetworkOperation(snapper, dt=snap_ms * ms)
            ops.net_ops.append(net_op)

        return ops

    # ── per-sample: collect data ──────────────────────────────────────────────

    def collect_sample(
        self,
        net: NetworkObjects,
        ops: SampleOps,
        T_steps: int,
        dt_ms: float = 1.0,
        duration_s: float | None = None,
        sim_duration_s: float | None = None,
    ):
        """
        Harvest data from monitors after brian2.run() has completed.
        Also performs L1 weight normalisation on dst neurons that spiked.
        Updates net.weights in-place with normalised matrices.
        """
        # ── Normalise weights ─────────────────────────────────────────────────
        arch_groups = self.arch.get("neuron_groups", {})
        arch_syns   = self.arch.get("synapse_groups", {})

        for sname, scfg in arch_syns.items():
            if scfg.get("plasticity", "none") not in ("stdp", "static"):
                continue
            _, dst_name = _parse_synapse_name(sname)

            norm_cfg   = scfg.get("norm", {})
            norm_limit = float(norm_cfg.get("limit", 0.0))
            on_group   = norm_cfg.get("on_group", "dst")   # 'dst' or 'src'
            wbounds    = scfg.get("weight_bounds", {})
            wmin       = float(wbounds.get("wmin", 0.0))
            wmax       = float(wbounds.get("wmax", 1.0))

            n_src = int(arch_groups[_parse_synapse_name(sname)[0]]["n"])
            n_dst = int(arch_groups[dst_name]["n"])

            W_new = np.zeros((n_src, n_dst))
            W_new[net.src_idx[sname], net.tgt_idx[sname]] = np.array(
                net.synapses[sname].w
            )

            # which neurons spiked in the normalisation reference group?
            norm_gname = dst_name if on_group == "dst" else _parse_synapse_name(sname)[0]
            if norm_gname in ops.spike_monitors and norm_limit > 0:
                spiked = np.unique(np.array(ops.spike_monitors[norm_gname].i))
                for nrn in spiked:
                    col_s = W_new[:, nrn].sum()
                    if col_s > norm_limit:
                        W_new[:, nrn] *= norm_limit / col_s

            W_new = np.clip(W_new, wmin, wmax)
            net.weights[sname] = W_new

        # ── Spikes ───────────────────────────────────────────────────────────
        for gname in (self._raster_groups | self._mfr_groups):
            if gname not in ops.spike_monitors:
                continue
            sm = ops.spike_monitors[gname]
            self.ep[f"spikes_{gname}_i"].append(np.array(sm.i,      dtype=np.int32))
            self.ep[f"spikes_{gname}_t"].append(np.array(sm.t / ms, dtype=np.float32))

        # ── Membrane potential ────────────────────────────────────────────────
        for gname, vm in ops.vmon.items():
            self.ep[f"vmon_{gname}_v"].append(np.array(vm.v,      dtype=np.float32))
            self.ep[f"vmon_{gname}_t"].append(np.array(vm.t / ms, dtype=np.float32))

        # ── Weight snapshot: per-sample arrays ───────────────────────────────
        for sname in self._we_cfg:
            if sname not in ops.snap_times:
                continue
            self.ep[f"we_{sname}_sample_times"].append(
                np.array(ops.snap_times[sname], dtype=np.float32)
            )
            self.ep[f"we_{sname}_sample_vals"].append(
                np.array(ops.snap_vals[sname],  dtype=np.float32)
            )

        # ── Final weight matrices ─────────────────────────────────────────────
        for sname in self._final_w_syns:
            if sname in net.weights:
                self.ep[f"w_{sname}_per_sample"].append(
                    net.weights[sname].astype(np.float32)
                )

        # ── Duration ─────────────────────────────────────────────────────────
        if duration_s is None:
            duration_s = T_steps * dt_ms * 1e-3
        if sim_duration_s is None:
            sim_duration_s = T_steps * dt_ms * 1e-3
        self.ep["sample_durations_s"].append(float(duration_s))
        self.ep["sample_sim_durations_s"].append(float(sim_duration_s))

    # ── epoch save ────────────────────────────────────────────────────────────

    def save_epoch(
        self,
        epoch_idx: int,
        out_dir: str,
        w_current: dict[str, np.ndarray],
    ) -> str:
        """
        Pack accumulated data and write history_epoch_NNN.npz.
        Returns the save path.
        """
        arrs: dict[str, Any] = {}

        # ── weight evolution ──────────────────────────────────────────────────
        for sname, pairs in self._we_cfg.items():
            arrs[f"we_{sname}_pairs"]        = np.array(pairs, dtype=np.int32)
            arrs[f"we_{sname}_times_ms"]     = np.array(
                self.ep[f"we_{sname}_times_ms"], dtype=np.float32)
            arrs[f"we_{sname}_values"]       = np.array(
                self.ep[f"we_{sname}_values"], dtype=np.float32)
            arrs[f"we_{sname}_sample_times"] = _pack_ragged(
                self.ep[f"we_{sname}_sample_times"])
            arrs[f"we_{sname}_sample_vals"]  = _pack_ragged(
                self.ep[f"we_{sname}_sample_vals"])
            arrs[f"we_{sname}_n_samples"]    = np.int32(
                len(self.ep[f"we_{sname}_sample_vals"]))

        # ── spikes ───────────────────────────────────────────────────────────
        arch_groups = self.arch.get("neuron_groups", {})
        for gname in (self._raster_groups | self._mfr_groups):
            key_i = f"spikes_{gname}_i"
            if not self.ep.get(key_i):
                continue
            n_nrn = int(arch_groups.get(gname, {}).get("n", 0))
            arrs[f"spikes_{gname}_i"]         = _pack_ragged(self.ep[key_i])
            arrs[f"spikes_{gname}_t"]         = _pack_ragged(self.ep[f"spikes_{gname}_t"])
            arrs[f"spikes_{gname}_n_samples"] = np.int32(len(self.ep[key_i]))
            arrs[f"spikes_{gname}_n_neurons"] = np.int32(n_nrn)
            arrs[f"spikes_{gname}_has_raster"]= np.bool_(gname in self._raster_groups)
            arrs[f"spikes_{gname}_has_mfr"]   = np.bool_(gname in self._mfr_groups)

        # ── membrane potential ────────────────────────────────────────────────
        for gname, (idx, windows) in self._vmon_cfg.items():
            key_v = f"vmon_{gname}_v"
            if not self.ep.get(key_v):
                continue
            arrs[f"vmon_{gname}_indices"]   = np.array(idx, dtype=np.int32)
            arrs[f"vmon_{gname}_v_all"]     = _pack_ragged(self.ep[key_v])
            arrs[f"vmon_{gname}_t_all"]     = _pack_ragged(self.ep[f"vmon_{gname}_t"])
            arrs[f"vmon_{gname}_n_samples"] = np.int32(len(self.ep[key_v]))
            win_rows = []
            for nid, w in zip(idx, windows):
                win_rows.append(
                    [nid, -1.0, -1.0] if w is None else [nid, float(w[0]), float(w[1])]
                )
            arrs[f"vmon_{gname}_windows"] = np.array(win_rows, dtype=np.float32)

        # ── final weight matrices ─────────────────────────────────────────────
        for sname in self._final_w_syns:
            key = f"w_{sname}_per_sample"
            if not self.ep.get(key):
                continue
            arrs[f"w_{sname}_final"]     = w_current.get(sname, np.array([])).astype(np.float32)
            arrs[f"w_{sname}_per_sample"] = _pack_ragged(self.ep[key])
            arrs[f"w_{sname}_n_samples"]  = np.int32(len(self.ep[key]))

        # ── metadata ──────────────────────────────────────────────────────────
        arrs["sample_durations_s"]     = np.array(
            self.ep["sample_durations_s"],     dtype=np.float32)
        arrs["sample_sim_durations_s"] = np.array(
            self.ep["sample_sim_durations_s"], dtype=np.float32)
        for gname, gcfg in arch_groups.items():
            arrs[f"N_{gname}"] = np.int32(gcfg["n"])

        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"history_epoch_{epoch_idx:03d}.npz")
        np.savez_compressed(save_path, **arrs)
        print(f"Saved epoch {epoch_idx} → {save_path}")
        return save_path

    # ── top-k logging ─────────────────────────────────────────────────────────

    def log_top_k(self, w_final: dict[str, np.ndarray], out_dir: str):
        """Write top-k weight logs to vizs/ directory."""
        top_k_cfg = self.cfg.get("top_k_weights") or {}
        viz_dir   = os.path.join(out_dir, "vizs")
        os.makedirs(viz_dir, exist_ok=True)

        for sname, k in top_k_cfg.items():
            if sname not in w_final:
                continue
            W     = w_final[sname]
            k     = int(k)
            w_flat = W.flatten()
            top_idx = np.argsort(-w_flat)[:k]
            top_val = w_flat[top_idx]
            n_cols  = W.shape[1]

            fname    = f"final_top_k_weights_{sname}.log"
            log_path = os.path.join(viz_dir, fname)
            with open(log_path, "w") as flog:
                flog.write(f"Top {k} weights after training ({sname}, sorted descending)\n")
                flog.write(f"{'Rank':>5}  {'src':>10}  {'dst':>10}  {'weight':>12}\n")
                flog.write("-" * 45 + "\n")
                for rank, (fidx, val) in enumerate(zip(top_idx, top_val), 1):
                    src_n = int(fidx) // n_cols
                    dst_n = int(fidx) %  n_cols
                    flog.write(f"{rank:>5}  {src_n:>10}  {dst_n:>10}  {val:>12.6f}\n")
            print(f"Saved top-k {sname} log → {log_path}")