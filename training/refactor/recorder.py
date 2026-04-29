"""
recorder.py — passive registry recorder. Never edit this file when adding groups.

All configuration lives in record_config.yaml.
All registration happens in train.py via:
    recorder.track_group("input",   G_in)
    recorder.track_group("hidden",  G_h)
    recorder.track_group("output",  G_out)   # adding a group = this one line
    recorder.track_synapses("in->hid",  S_ih, src_ih, tgt_ih)
    recorder.track_synapses("hid->out", S_ho, src_ho, tgt_ho)
"""

import os
import numpy as np
import yaml
from brian2 import SpikeMonitor, StateMonitor, network_operation, ms


def _pack_ragged(lst):
    out = np.empty(len(lst), dtype=object)
    for i, a in enumerate(lst):
        out[i] = a
    return out


def _parse_vmon_entries(raw):
    indices, windows = [], []
    for entry in (raw or []):
        if isinstance(entry, (int, float)):
            nid = int(entry)
            indices.append(nid)
            windows.append((nid, None))
        else:
            nid = int(entry[0])
            win = entry[1] if len(entry) > 1 else None
            indices.append(nid)
            windows.append((nid, win))
    return indices, windows


class Recorder:
    """
    Passive registry recorder.

    Workflow
    --------
    1.  recorder = Recorder("record_config.yaml", net)
    2.  recorder.track_group(name, group)       — one call per NeuronGroup
    3.  recorder.track_synapses(name, syn, src, tgt)  — one call per Synapses
    4.  recorder.build()                        — call once after all registrations
    5.  Loop:
            recorder.before_sample(sample_idx)
            net.run(T * DT_SIM)
            recorder.after_sample(sample_idx, duration_s, w_matrices={...})
    6.  recorder.save_epoch(epoch_idx, save_dir, final_weights={...})
        recorder.reset_epoch()
    """

    def __init__(self, config_path, net):
        self._net      = net
        self._groups   = {}   # name -> {group, cfg, mon, _spike_baseline}
        self._synapses = {}   # name -> {syn, src, tgt, cfg, mon}
        self._built    = False

        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._snapshot_interval = self._cfg.get("snapshot_interval_ms", 500) * ms

    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────

    def track_group(self, name, group):
        """
        Register a NeuronGroup.
        'name' must match the key under 'groups:' in record_config.yaml.
        If the name is absent from the config nothing is recorded — safe to
        register groups you don't currently need to record.
        """
        gcfg = self._cfg.get("groups", {}).get(name, {})
        self._groups[name] = {"group": group, "cfg": gcfg, "mon": {}}

    def track_synapses(self, name, syn, src_array, tgt_array):
        """
        Register a Synapses object.
        'name' must match the key under 'synapses:' in record_config.yaml.
        src_array / tgt_array are np.array(S.i) / np.array(S.j) pre-computed
        before the training loop.
        """
        scfg = self._cfg.get("synapses", {}).get(name, {})
        self._synapses[name] = {
            "syn": syn, "src": src_array, "tgt": tgt_array,
            "cfg": scfg, "mon": {}
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Build  (once, after all registrations)
    # ─────────────────────────────────────────────────────────────────────────

    def build(self):
        """Attach Brian2 monitors based on config + registered objects."""
        assert not self._built, "build() called twice"

        for name, entry in self._groups.items():
            g   = entry["group"]
            cfg = entry["cfg"]
            mon = entry["mon"]

            # Spike monitor — used for raster and/or mean firing rate
            if cfg.get("spike_raster") or cfg.get("mean_firing_rate"):
                mon["spike"] = SpikeMonitor(g)
                self._net.add(mon["spike"])

            # Voltage monitor — variables and neuron indices set per-group in config
            vmon_raw = cfg.get("membrane_potential", [])
            if vmon_raw:
                indices, windows = _parse_vmon_entries(vmon_raw)
                variables = cfg.get("membrane_potential_variables", ["v"])
                mon["voltage"]        = StateMonitor(g, variables, record=indices)
                mon["vmon_indices"]   = indices
                mon["vmon_windows"]   = windows
                mon["vmon_variables"] = variables
                self._net.add(mon["voltage"])

        for name, entry in self._synapses.items():
            cfg = entry["cfg"]
            mon = entry["mon"]
            syn = entry["syn"]
            src = entry["src"]
            tgt = entry["tgt"]

            evolution_pairs = cfg.get("weights_evolution", [])
            if evolution_pairs:
                pair_flat_idx = []
                for (pi, pj) in evolution_pairs:
                    mask = (src == pi) & (tgt == pj)
                    idx  = np.where(mask)[0]
                    pair_flat_idx.append(int(idx[0]) if len(idx) > 0 else None)

                mon["we_pairs"]         = evolution_pairs
                mon["we_pair_flat_idx"] = pair_flat_idx
                mon["we_snap_times"]    = []
                mon["we_snap_values"]   = [[] for _ in evolution_pairs]

                # Brian2 network_operation only accepts (t) or () — no extra args.
                # Capture entry via a factory function to avoid the late-binding
                # closure problem when multiple synapse groups are registered.
                def _make_snap_op(e, interval):
                    @network_operation(dt=interval)
                    def _snap(t):
                        m     = e["mon"]
                        t_ms  = float(t / ms)
                        w_arr = np.array(e["syn"].w)
                        m["we_snap_times"].append(t_ms)
                        for k, fidx in enumerate(m["we_pair_flat_idx"]):
                            val = float(w_arr[fidx]) if fidx is not None else float("nan")
                            m["we_snap_values"][k].append(val)
                    return _snap

                _snap = _make_snap_op(entry, self._snapshot_interval)

                mon["we_snap_op"] = _snap
                self._net.add(_snap)

        self._built = True
        self.reset_epoch()

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch control
    # ─────────────────────────────────────────────────────────────────────────

    def reset_epoch(self):
        self._elapsed_ms = 0.0
        self._epoch = {"groups": {}, "synapses": {}}

        for name, entry in self._groups.items():
            cfg = entry["cfg"]
            n   = entry["group"].N
            e   = {}
            if cfg.get("spike_raster"):
                e["raster_i"] = []
                e["raster_t"] = []
            if cfg.get("mean_firing_rate"):
                e["mfr_counts"]        = np.zeros(n, dtype=np.int64)
                e["mfr_dur_s"]         = 0.0
                e["mfr_sample_counts"] = []
                e["mfr_sample_dur_s"]  = []
            if entry["mon"].get("voltage") is not None:
                e["vmon_t_all"]  = []
                e["vmon_v_all"]  = {var: [] for var in entry["mon"].get("vmon_variables", [])}
            self._epoch["groups"][name] = e

        for name, entry in self._synapses.items():
            cfg = entry["cfg"]
            e   = {}
            if cfg.get("weights_evolution"):
                e["we_times_ms"]      = []
                e["we_values"]        = [[] for _ in cfg["weights_evolution"]]
                e["we_sample_times"]  = []
                e["we_sample_values"] = []
            if cfg.get("final_weights_per_sample"):
                e["weight_per_sample"] = []
            self._epoch["synapses"][name] = e

    # ─────────────────────────────────────────────────────────────────────────
    # Per-sample hooks
    # ─────────────────────────────────────────────────────────────────────────

    def before_sample(self, sample_idx):
        """Snapshot spike-monitor lengths so we can isolate this sample's spikes."""
        _ = sample_idx
        self._sample_start_ms = self._elapsed_ms
        for entry in self._groups.values():
            mon = entry["mon"]
            if "spike" in mon:
                entry["_spike_baseline"] = len(mon["spike"].t)

        for entry in self._synapses.values():
            mon = entry["mon"]
            if "we_snap_times" in mon:
                mon["we_snap_times"]  = []
                mon["we_snap_values"] = [[] for _ in mon["we_pairs"]]

    def after_sample(self, sample_idx, duration_s, w_matrices=None):
        """
        Collect monitor data after net.run().

        Parameters
        ----------
        sample_idx  : int
        duration_s  : float
        w_matrices  : dict[synapse_name -> np.ndarray (N_pre, N_post)] or None
                      Post-normalisation weight matrix for each synapse group
                      you want recorded per-sample.
                      e.g. {"in->hid": w_ih, "hid->out": w_ho}
        """
        _ = sample_idx
        w_matrices = w_matrices or {}

        start_ms = getattr(self, "_sample_start_ms", 0.0)

        def _slice(mon, baseline):
            i_all = np.array(mon.i)
            t_all = np.array(mon.t / ms) - start_ms
            return i_all[baseline:], t_all[baseline:]

        for name, entry in self._groups.items():
            cfg      = entry["cfg"]
            mon      = entry["mon"]
            e        = self._epoch["groups"][name]
            n        = entry["group"].N
            baseline = entry.get("_spike_baseline", 0)

            if "spike" in mon:
                i_s, t_s = _slice(mon["spike"], baseline)

                if cfg.get("spike_raster"):
                    e["raster_i"].append(i_s.astype(np.int32))
                    e["raster_t"].append(t_s.astype(np.float32))

                if cfg.get("mean_firing_rate"):
                    counts = (np.bincount(i_s, minlength=n).astype(np.int64)
                              if len(i_s) > 0 else np.zeros(n, dtype=np.int64))
                    e["mfr_counts"] += counts
                    e["mfr_dur_s"]  += duration_s
                    e["mfr_sample_counts"].append(counts.astype(np.int32))
                    e["mfr_sample_dur_s"].append(duration_s)

            if "voltage" in mon:
                vmon = mon["voltage"]
                for var in mon["vmon_variables"]:
                    e["vmon_v_all"][var].append(
                        np.array(getattr(vmon, var), dtype=np.float32)
                    )
                e["vmon_t_all"].append(np.array(vmon.t / ms, dtype=np.float32))

        for name, entry in self._synapses.items():
            cfg = entry["cfg"]
            mon = entry["mon"]
            e   = self._epoch["synapses"][name]

            if "we_snap_times" in mon:
                e["we_times_ms"].extend(mon["we_snap_times"])
                for k, vals in enumerate(mon["we_snap_values"]):
                    e["we_values"][k].extend(vals)
                e["we_sample_times"].append(
                    np.array(mon["we_snap_times"],  dtype=np.float32)
                )
                e["we_sample_values"].append(
                    np.array(mon["we_snap_values"], dtype=np.float32)
                )

            if cfg.get("final_weights_per_sample") and name in w_matrices:
                e["weight_per_sample"].append(w_matrices[name].astype(np.float32))

        self._elapsed_ms += duration_s * 1000.0

    def spikes_this_sample(self, group_name):
        """
        Return spike neuron-indices for a group from the current sample only.
        Use this in train.py for normalisation that depends on which neurons fired.
        """
        entry    = self._groups[group_name]
        mon      = entry["mon"]
        baseline = entry.get("_spike_baseline", 0)
        if "spike" not in mon:
            return np.array([], dtype=np.int32)
        return np.array(mon["spike"].i)[baseline:]

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def save_epoch(self, epoch_idx, save_dir, final_weights=None, init_weights=None):
        """
        Pack and save epoch data to a compressed .npz.

        Parameters
        ----------
        epoch_idx     : int
        save_dir      : str
        final_weights : dict[synapse_name -> np.ndarray] or None
        init_weights  : dict[synapse_name -> np.ndarray] or None  (epoch 0 only)
        """
        arrays        = {}
        final_weights = final_weights or {}
        init_weights  = init_weights  or {}

        # Array keys are prefixed with the group/synapse name so nothing collides
        for name, entry in self._groups.items():
            cfg = entry["cfg"]
            mon = entry["mon"]
            e   = self._epoch["groups"][name]
            n   = entry["group"].N
            pfx = name.replace("->", "_to_")

            if cfg.get("spike_raster") and e.get("raster_i"):
                arrays[f"{pfx}__raster_i"]         = _pack_ragged(e["raster_i"])
                arrays[f"{pfx}__raster_t"]         = _pack_ragged(e["raster_t"])
                arrays[f"{pfx}__raster_n_samples"] = np.int32(len(e["raster_i"]))
                arrays[f"{pfx}__raster_n_neurons"] = np.int32(n)

            if cfg.get("mean_firing_rate") and e.get("mfr_dur_s", 0) > 0:
                dur = e["mfr_dur_s"]
                arrays[f"{pfx}__mfr"]               = (e["mfr_counts"] / dur).astype(np.float32)
                arrays[f"{pfx}__mfr_sample_counts"] = _pack_ragged(e["mfr_sample_counts"])
                arrays[f"{pfx}__mfr_sample_dur_s"]  = np.array(e["mfr_sample_dur_s"], dtype=np.float32)

            if mon.get("voltage") is not None and e.get("vmon_t_all"):
                arrays[f"{pfx}__vmon_indices"] = np.array(mon["vmon_indices"], dtype=np.int32)
                arrays[f"{pfx}__vmon_t_all"]   = _pack_ragged(e["vmon_t_all"])
                for var in mon["vmon_variables"]:
                    arrays[f"{pfx}__vmon_{var}_all"] = _pack_ragged(e["vmon_v_all"][var])
                win_rows = []
                for nid, w in mon["vmon_windows"]:
                    win_rows.append([nid, float(w[0]), float(w[1])] if w else [nid, -1., -1.])
                arrays[f"{pfx}__vmon_windows"] = np.array(win_rows, dtype=np.float32)

        for name, entry in self._synapses.items():
            cfg = entry["cfg"]
            e   = self._epoch["synapses"][name]
            pfx = name.replace("->", "_to_")

            if cfg.get("weights_evolution") and e.get("we_times_ms"):
                arrays[f"{pfx}__we_pairs"]         = np.array(cfg["weights_evolution"], dtype=np.int32)
                arrays[f"{pfx}__we_values"]        = np.array(e["we_values"],   dtype=np.float32)
                arrays[f"{pfx}__we_times_ms"]      = np.array(e["we_times_ms"], dtype=np.float32)
                arrays[f"{pfx}__we_sample_times"]  = _pack_ragged(e["we_sample_times"])
                arrays[f"{pfx}__we_sample_values"] = _pack_ragged(e["we_sample_values"])

            if cfg.get("final_weights_per_sample") and e.get("weight_per_sample"):
                arrays[f"{pfx}__weight_per_sample"] = _pack_ragged(e["weight_per_sample"])
                arrays[f"{pfx}__weight_n_samples"]  = np.int32(len(e["weight_per_sample"]))

            if name in final_weights:
                arrays[f"{pfx}__final_weights"] = final_weights[name].astype(np.float32)

            if name in init_weights:
                arrays[f"{pfx}__init_weights"] = init_weights[name].astype(np.float32)

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"history_epoch_{epoch_idx:03d}.npz")
        np.savez_compressed(path, **arrays)
        print(f"  [recorder] saved epoch {epoch_idx} → {path}")