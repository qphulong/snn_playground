"""
network_builder.py
------------------
Reads architecture.yaml and constructs Brian2 NeuronGroups, Synapses,
and initial weight matrices.

Public API
----------
    arch   = load_architecture(path)          # parse + validate YAML
    result = build_network(arch, I_ext_array) # returns NetworkObjects

NetworkObjects (dataclass)
--------------------------
    .groups   : dict[str, NeuronGroup]
    .synapses : dict[str, Synapses]
    .weights  : dict[str, np.ndarray]   # shape (N_src, N_dst), current values
    .src_idx  : dict[str, np.ndarray]   # flat synapse source indices
    .tgt_idx  : dict[str, np.ndarray]   # flat synapse target indices
    .arch     : dict                    # original parsed architecture dict

Design notes
------------
- Neuron equations are taken verbatim from the YAML.
  The special placeholder `I_ext` is a Brian2 TimedArray that the caller
  must supply; it is injected into the Brian2 namespace automatically.
- Weight init: only circular_gaussian is currently supported.
- Synapse names must follow: from_<src>_to_<dst>.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
from brian2 import (
    NeuronGroup,
    Synapses,
    TimedArray,
    ms,
    defaultclock,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_synapse_name(name: str) -> tuple[str, str]:
    """Extract (src_group, dst_group) from 'from_X_to_Y'.
    Handles group names that themselves contain underscores.
    Strategy: split on '_to_' (right-most occurrence after 'from_').
    """
    m = re.fullmatch(r"from_(.+)_to_(.+)", name)
    if m is None:
        raise ValueError(
            f"Synapse group name '{name}' does not match pattern "
            f"'from_<src>_to_<dst>'. Please rename it."
        )
    return m.group(1), m.group(2)


def _circular_gaussian(
    n_src: int,
    n_dst: int,
    sigma_frac: float = 0.2,
    init_sum: float = 2.0,
    noise_std: float = 0.005,
    wmin: float = 0.0,
    wmax: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Circular-Gaussian weight matrix, L1-normalised per column to init_sum."""
    if rng is None:
        rng = np.random.default_rng()
    i_idx   = np.arange(n_src).reshape(-1, 1)
    j_scaled = np.arange(n_dst).reshape(1, -1) * (n_src / n_dst)
    dist    = np.abs(i_idx - j_scaled)
    dist    = np.minimum(dist, n_src - dist)
    sigma   = n_src * sigma_frac
    W       = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    W      += rng.normal(0, noise_std, size=W.shape)
    W       = np.clip(W, 0, None)
    col_sums = W.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    W = W / col_sums * init_sum
    return np.clip(W, wmin, wmax).astype(np.float64)


# ── public data container ─────────────────────────────────────────────────────

@dataclass
class NetworkObjects:
    """All Brian2 objects produced by build_network()."""
    groups:   dict[str, NeuronGroup]        = field(default_factory=dict)
    synapses: dict[str, Synapses]           = field(default_factory=dict)
    weights:  dict[str, np.ndarray]         = field(default_factory=dict)
    src_idx:  dict[str, np.ndarray]         = field(default_factory=dict)
    tgt_idx:  dict[str, np.ndarray]         = field(default_factory=dict)
    arch:     dict                          = field(default_factory=dict)


# ── YAML loading & validation ─────────────────────────────────────────────────

def load_architecture(path: str) -> dict:
    """Parse architecture.yaml and perform basic validation."""
    with open(path, "r") as f:
        arch = yaml.safe_load(f)

    groups   = arch.get("neuron_groups", {})
    synapses = arch.get("synapse_groups", {})

    if not groups:
        raise ValueError("architecture.yaml: 'neuron_groups' is empty or missing.")

    input_groups = [k for k, v in groups.items() if v.get("is_input", False)]
    if not input_groups:
        raise ValueError(
            "architecture.yaml: at least one neuron group must have 'is_input: true'."
        )

    for syn_name in synapses:
        src, dst = _parse_synapse_name(syn_name)  # raises on bad name
        if src not in groups:
            raise ValueError(
                f"Synapse '{syn_name}': source group '{src}' not found in neuron_groups."
            )
        if dst not in groups:
            raise ValueError(
                f"Synapse '{syn_name}': destination group '{dst}' not found in neuron_groups."
            )

    return arch


# ── network builder ───────────────────────────────────────────────────────────

def build_network(
    arch: dict,
    I_ext_array: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> NetworkObjects:
    """
    Build Brian2 NeuronGroups and Synapses from arch dict.

    Parameters
    ----------
    arch        : parsed architecture dict (from load_architecture)
    I_ext_array : shape (T, N_input) float array — the spike-encoded input
                  current.  Converted internally to a TimedArray named I_ext.
    rng         : optional numpy Generator for reproducible weight init.

    Returns
    -------
    NetworkObjects with all Brian2 objects and weight arrays.
    """
    if rng is None:
        seed = arch.get("simulation", {}).get("random_seed", 42)
        rng  = np.random.default_rng(seed)

    dt_ms  = float(arch.get("simulation", {}).get("dt_ms", 1.0))
    dt_val = dt_ms * ms

    # TimedArray for external input (shape must be T × N)
    I_ext = TimedArray(I_ext_array.astype(float), dt=dt_val)  # noqa: F841
    # Brian2 picks this up from the local namespace in NeuronGroup equations
    # via the 'namespace' kwarg below.

    net = NetworkObjects(arch=arch)

    group_cfg = arch.get("neuron_groups", {})
    syn_cfg   = arch.get("synapse_groups", {})

    # ── Build NeuronGroups ────────────────────────────────────────────────────
    for gname, gcfg in group_cfg.items():
        n          = int(gcfg["n"])
        eqs        = gcfg["equations"]
        threshold  = gcfg["threshold"]
        reset      = gcfg["reset"]
        refractory = gcfg.get("refractory", "0*ms")
        method     = gcfg.get("method", "euler")

        kwargs: dict[str, Any] = dict(
            threshold  = threshold,
            reset      = reset,
            refractory = refractory,
            method     = method,
        )

        # Input groups need I_ext in their namespace
        if gcfg.get("is_input", False):
            kwargs["namespace"] = {"I_ext": I_ext}

        grp = NeuronGroup(n, eqs, **kwargs)
        for var, val in (gcfg.get("init") or {}).items():
            setattr(grp, var, float(val))
        net.groups[gname] = grp

    # ── Build Synapses ────────────────────────────────────────────────────────
    for sname, scfg in syn_cfg.items():
        src_name, dst_name = _parse_synapse_name(sname)
        G_src = net.groups[src_name]
        G_dst = net.groups[dst_name]

        n_src = int(group_cfg[src_name]["n"])
        n_dst = int(group_cfg[dst_name]["n"])

        plasticity = scfg.get("plasticity", "none")
        wbounds    = scfg.get("weight_bounds", {})
        wmin       = float(wbounds.get("wmin", 0.0))
        wmax       = float(wbounds.get("wmax", 1.0))

        # ── Weight init ───────────────────────────────────────────────────────
        wi_cfg    = scfg.get("weight_init", {})
        wi_method = wi_cfg.get("method", "circular_gaussian")
        if wi_method == "circular_gaussian":
            W_init = _circular_gaussian(
                n_src,
                n_dst,
                sigma_frac = float(wi_cfg.get("sigma_frac", 0.2)),
                init_sum   = float(wi_cfg.get("init_sum",   2.0)),
                noise_std  = float(wi_cfg.get("noise_std",  0.005)),
                wmin       = wmin,
                wmax       = wmax,
                rng        = rng,
            )
        else:
            raise NotImplementedError(
                f"Weight init method '{wi_method}' is not implemented. "
                f"Currently supported: circular_gaussian."
            )

        net.weights[sname] = W_init

        # ── Build Synapses object ─────────────────────────────────────────────
        if plasticity == "stdp":
            stdp_cfg = scfg["stdp"]
            syn = Synapses(
                G_src, G_dst,
                model   = stdp_cfg["model"],
                on_pre  = stdp_cfg["on_pre"].strip(),
                on_post = stdp_cfg["on_post"].strip(),
            )
        elif plasticity == "static":
            syn = Synapses(G_src, G_dst, model="w : 1", on_pre="v_post += w")
        else:  # none
            syn = Synapses(G_src, G_dst, on_pre="v_post += 1")

        syn.connect()  # all-to-all

        # Store flat index arrays before assigning weights
        src_arr = np.array(syn.i)
        tgt_arr = np.array(syn.j)
        net.src_idx[sname] = src_arr
        net.tgt_idx[sname] = tgt_arr

        # Assign initial weights via flat index
        if plasticity in ("stdp", "static"):
            syn.w = W_init[src_arr, tgt_arr]

        net.synapses[sname] = syn

    return net


# ── weight extraction helper (used by train.py after each sample) ─────────────

def extract_weights(net: NetworkObjects) -> dict[str, np.ndarray]:
    """Pull current weight values out of Brian2 synapses → 2-D matrices."""
    result = {}
    arch_groups = net.arch.get("neuron_groups", {})
    arch_syns   = net.arch.get("synapse_groups", {})

    for sname in net.synapses:
        scfg = arch_syns.get(sname, {})
        if scfg.get("plasticity", "none") not in ("stdp", "static"):
            continue
        src_name, dst_name = _parse_synapse_name(sname)
        n_src = int(arch_groups[src_name]["n"])
        n_dst = int(arch_groups[dst_name]["n"])

        W = np.zeros((n_src, n_dst))
        W[net.src_idx[sname], net.tgt_idx[sname]] = np.array(net.synapses[sname].w)
        result[sname] = W

    return result


def inject_weights(net: NetworkObjects, weights: dict[str, np.ndarray]) -> None:
    """Push 2-D weight matrices back into Brian2 synapses (used between samples)."""
    for sname, W in weights.items():
        if sname not in net.synapses:
            continue
        scfg = net.arch.get("synapse_groups", {}).get(sname, {})
        if scfg.get("plasticity", "none") not in ("stdp", "static"):
            continue
        net.synapses[sname].w = W[net.src_idx[sname], net.tgt_idx[sname]]