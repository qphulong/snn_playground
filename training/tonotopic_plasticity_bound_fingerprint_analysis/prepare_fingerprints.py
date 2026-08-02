"""
prepare_fingerprints.py
=======================
Trains independent SNN runs (one per speaker/recording/part triplet) using the
tonotopic_plasticity_bound architecture and saves weight fingerprints.

Audio is supplied via two nested dicts at the top of this file:
  WAV_FILES_A[person][session] = path   → part "A" fingerprints (heatmap rows)
  WAV_FILES_B[person][session] = path   → part "B" fingerprints (heatmap cols)
Layout: 5 persons × 3 sessions each × 2 parts (A, B) × 1 audio file per part.
Each (part, person, session) is trained into one fingerprint from its single file.
A and B must share the same persons+sessions so heatmap.py's speaker blocks align.

Training: NUM_EPOCHS per run.  Weight snapshots are collected inside
normalize_weights() every 100 ms for epochs >= SNAPSHOT_FROM_EPOCH.
The fingerprint is the mean of all collected snapshots (float32).

All runs share the same initial weight matrices (seed 42) so differences between
fingerprints reflect audio content, not random initialisation.

Output: fingerprints.npz
  fingerprints  (N, 192, 192)  float32  — averaged weight matrices
  person_ids    (N,)           str
  record_ids    (N,)           str
  parts         (N,)           str      — "A" or "B"
  wav_files     (N,)           object   — list of paths per fingerprint

Usage:
  cd <repo_root>
  python training/tonotopic_plasticity_bound_fingerprint_analysis/prepare_fingerprints.py
"""

import numpy as np
np.random.seed(42)
import os
import sys
import time

from brian2 import (
    NeuronGroup, Synapses, TimedArray, Network, network_operation,
    defaultclock, ms, second,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, '..', '..')

sys.path.insert(0, REPO_ROOT)
from src.utils.spike_encoding import compute_spike_input_current

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

OUT_PATH     = os.path.join(SCRIPT_DIR, 'fingerprints.npz')

# ─────────────────────────────────────────────────────────────────────────────
# Audio inputs — fill in manually.  Structure:  WAV_FILES_{A,B}[person][session] = path
#
#   5 persons × 3 sessions each × 2 parts (A, B) × 1 audio file per part.
#   Each (part, person, session) → one fingerprint trained on that single file.
#
# WAV_FILES_A and WAV_FILES_B MUST have identical persons+sessions so the part-A
# (rows) and part-B (cols) fingerprints line up speaker-block by speaker-block in
# heatmap.py (BLOCK = 3 sessions per speaker → 5 speakers).
# Paths may be absolute or relative to the repo root (cwd = repo root when run).
# ─────────────────────────────────────────────────────────────────────────────

# Part A → 00001.m4a of each session; Part B → 00002.m4a of the SAME session.
WAV_FILES_A = {
    "id10604": {
        "0yJ2UKJfCSM": "datasets/vox1/dev_04/id10604/0yJ2UKJfCSM/00001.m4a",
        "4tlCKgb3LvU": "datasets/vox1/dev_04/id10604/4tlCKgb3LvU/00001.m4a",
        "98kmA6XwM9U": "datasets/vox1/dev_04/id10604/98kmA6XwM9U/00001.m4a",
    },
    "id11005": {
        "4Sw6odb03ig": "datasets/vox1/dev_06/id11005/4Sw6odb03ig/00001.m4a",
        "6DTGOCa0qk0": "datasets/vox1/dev_06/id11005/6DTGOCa0qk0/00001.m4a",
        "6YT0_aBfQI0": "datasets/vox1/dev_06/id11005/6YT0_aBfQI0/00001.m4a",
    },
    "id10204": {
        "EKIF0zKoyss": "datasets/vox1/dev_02/id10204/EKIF0zKoyss/00001.m4a",
        "GO4hi9004hg": "datasets/vox1/dev_02/id10204/GO4hi9004hg/00001.m4a",
        "Hu9TdJRQYYo": "datasets/vox1/dev_02/id10204/Hu9TdJRQYYo/00001.m4a",
    },
    "id10257": {
        "1uqrS4yUk34": "datasets/vox1/dev_02/id10257/1uqrS4yUk34/00001.m4a",
        "K15lRgcitSw": "datasets/vox1/dev_02/id10257/K15lRgcitSw/00001.m4a",
        "Ko-rW5RWuOo": "datasets/vox1/dev_02/id10257/Ko-rW5RWuOo/00001.m4a",
    },
    "id11248": {
        "4iN6W9ajA-g": "datasets/vox1/dev_07/id11248/4iN6W9ajA-g/00001.m4a",
        "5hBbAqRV_uc": "datasets/vox1/dev_07/id11248/5hBbAqRV_uc/00001.m4a",
        "6kGwhudccdg": "datasets/vox1/dev_07/id11248/6kGwhudccdg/00001.m4a",
    },
}

WAV_FILES_B = {
    "id10604": {
        "0yJ2UKJfCSM": "datasets/vox1/dev_04/id10604/0yJ2UKJfCSM/00002.m4a",
        "4tlCKgb3LvU": "datasets/vox1/dev_04/id10604/4tlCKgb3LvU/00002.m4a",
        "98kmA6XwM9U": "datasets/vox1/dev_04/id10604/98kmA6XwM9U/00002.m4a",
    },
    "id11005": {
        "4Sw6odb03ig": "datasets/vox1/dev_06/id11005/4Sw6odb03ig/00002.m4a",
        "6DTGOCa0qk0": "datasets/vox1/dev_06/id11005/6DTGOCa0qk0/00002.m4a",
        "6YT0_aBfQI0": "datasets/vox1/dev_06/id11005/6YT0_aBfQI0/00002.m4a",
    },
    "id10204": {
        "EKIF0zKoyss": "datasets/vox1/dev_02/id10204/EKIF0zKoyss/00002.m4a",
        "GO4hi9004hg": "datasets/vox1/dev_02/id10204/GO4hi9004hg/00002.m4a",
        "Hu9TdJRQYYo": "datasets/vox1/dev_02/id10204/Hu9TdJRQYYo/00002.m4a",
    },
    "id10257": {
        "1uqrS4yUk34": "datasets/vox1/dev_02/id10257/1uqrS4yUk34/00002.m4a",
        "K15lRgcitSw": "datasets/vox1/dev_02/id10257/K15lRgcitSw/00002.m4a",
        "Ko-rW5RWuOo": "datasets/vox1/dev_02/id10257/Ko-rW5RWuOo/00002.m4a",
    },
    "id11248": {
        "4iN6W9ajA-g": "datasets/vox1/dev_07/id11248/4iN6W9ajA-g/00002.m4a",
        "5hBbAqRV_uc": "datasets/vox1/dev_07/id11248/5hBbAqRV_uc/00002.m4a",
        "6kGwhudccdg": "datasets/vox1/dev_07/id11248/6kGwhudccdg/00002.m4a",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (match training/tonotopic_plasticity_bound/train.py)
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 192   # 64 channels × 3 neurons/channel
N_H  = 192

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 1
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF) --
tau_h    = 50 * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

# -- Soft refractory (hidden only) --
tau_r = 10 * ms

# -- Membrane noise (hidden only) --
sigma_noise = 0.03 * second**(-0.5)

# -- STDP (excitatory) --
taupre  = 20 * ms
taupost = 20 * ms

# -- Triplet STDP (excitatory; Pfister-Gerstner, added on top of the pair rule) --
# Slow detector traces r2 (pre) and o2 (post) make LTP/LTD depend on a recent
# *previous* spike of the same kind (frequency-dependent, BCM-like).
# Set both A3 centres to 0 to recover the plain pair-STDP behaviour exactly.
tau_x         = 100 * ms   # slow presynaptic detector r2
tau_y         = 125 * ms   # slow postsynaptic detector o2
A3PRE_CENTER  =  0.004     # triplet LTP amplitude (pre-post-post), > 0
A3POST_CENTER = -0.002     # triplet LTD amplitude (post-pre-pre), < 0

# -- Excitatory weight bounds --
wmin = 0.0

# -- Excitatory synapse --
WMAX_CENTER  = 1.0
APRE_CENTER  =  0.004
APOST_CENTER = -0.0048

# -- Inhibitory lateral synapse --
W_INH_CENTER = 1.0
W_INH_MIN    = 0.0
APRE_INH     = 0.004
APOST_INH    = -0.0048

# -- Channel layout --
N_CHANNELS    = 64
N_PER_CHANNEL = N_IN // N_CHANNELS   # 4

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d_channel/R)^p)) --
R_EXC_CHANNEL = 11
p_EXC         = 3

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 1.0455
NORM_LIMIT_INH = 0.4656

# -- Fingerprint collection --
NUM_EPOCHS         = 8
SNAPSHOT_FROM_EPOCH = 4   # collect snapshots from this epoch onward (0-indexed)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity matrices — excitatory
# Distance is measured in channels: d_ch = |ch_i - ch_j| (circular on N_CHANNELS).
# topo(d_ch) = max(0, 1-(d_ch/R_EXC_CHANNEL)^p); hard cutoff at d_ch=R_EXC_CHANNEL.
# ─────────────────────────────────────────────────────────────────────────────

_ch_i    = (np.arange(N_IN) // N_PER_CHANNEL).reshape(-1, 1)
_ch_j    = (np.arange(N_H)  // N_PER_CHANNEL).reshape(1, -1)
_dist_ch = np.abs(_ch_i - _ch_j)
_dist_ch = np.minimum(_dist_ch, N_CHANNELS - _dist_ch)
_topo_exc = np.maximum(0.0, 1.0 - (_dist_ch / R_EXC_CHANNEL) ** p_EXC)
_mask_ih  = _dist_ch <= R_EXC_CHANNEL
_src_ih, _tgt_ih = np.where(_mask_ih)

wmax_matrix   = WMAX_CENTER   * _topo_exc
Apre_matrix   = APRE_CENTER   * _topo_exc
Apost_matrix  = APOST_CENTER  * _topo_exc
A3pre_matrix  = A3PRE_CENTER  * _topo_exc   # triplet LTP amplitude (topo-scaled)
A3post_matrix = A3POST_CENTER * _topo_exc   # triplet LTD amplitude (topo-scaled)
del _ch_i, _ch_j, _dist_ch, _topo_exc

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute inhibitory plasticity matrices — Jaccard similarity.
# Two hidden neurons inhibit each other iff their excitatory input sets overlap.
# wmax/Apre/Apost scale by Jaccard = overlap_ch / (window_size + d_ch_hh).
# No self-connections.
# ─────────────────────────────────────────────────────────────────────────────

_ch_h       = np.arange(N_H) // N_PER_CHANNEL
_dist_ch_hh = np.abs(_ch_h.reshape(-1, 1) - _ch_h.reshape(1, -1))
_dist_ch_hh = np.minimum(_dist_ch_hh, N_CHANNELS - _dist_ch_hh)

_window_size = 2 * R_EXC_CHANNEL + 1
_overlap_ch  = np.maximum(0, _window_size - _dist_ch_hh)
_jaccard     = np.where(
    _overlap_ch > 0,
    _overlap_ch / (_window_size + _dist_ch_hh),
    0.0,
)
_mask_hh  = (_overlap_ch > 0) & (~np.eye(N_H, dtype=bool))
_src_hh, _tgt_hh = np.where(_mask_hh)

wmax_inh_matrix  = W_INH_CENTER * _jaccard
Apre_inh_matrix  = APRE_INH     * _jaccard
Apost_inh_matrix = APOST_INH    * _jaccard
del _ch_h, _dist_ch_hh, _overlap_ch, _jaccard

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrices — same for all runs
# Excitatory: formula-shaped, column-normalised to NORM_LIMIT_EXC
# Inhibitory: uniform random init in [0.01, 0.02] on connected positions
#             (seeded at module load, so identical across all runs)
# ─────────────────────────────────────────────────────────────────────────────

W_IH_INIT = np.zeros((N_IN, N_H))
W_IH_INIT[_src_ih, _tgt_ih] = wmax_matrix[_src_ih, _tgt_ih]
for _j in range(N_H):
    _col_mask = (_tgt_ih == _j)
    _rows = _src_ih[_col_mask]
    _wsum = W_IH_INIT[_rows, _j].sum()
    if _wsum > 0:
        W_IH_INIT[_rows, _j] *= NORM_LIMIT_EXC / _wsum

W_HH_INIT = np.zeros((N_H, N_H))
W_HH_INIT[_src_hh, _tgt_hh] = np.random.uniform(0.01, 0.02, size=_src_hh.shape[0])

# ─────────────────────────────────────────────────────────────────────────────
# Build Brian2 network (once — shared across all runs)
# ─────────────────────────────────────────────────────────────────────────────

defaultclock.dt = DT_SIM

# ── Input neurons ──────────────────────────────────────────────────────────────
eqs_in = """
dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
da/dt = -a / tau_a : 1
"""
G_in = NeuronGroup(
    N_IN, eqs_in,
    threshold="v > v_th_in",
    reset="v=0; a+=beta",
    refractory=2 * ms,
    method="euler"
)
_dummy_I = np.zeros((1, N_IN), dtype=float)
G_in.namespace["I_timed"] = TimedArray(_dummy_I, dt=DT_SIM)

# ── Hidden neurons ─────────────────────────────────────────────────────────────
eqs_h = f"""
dv/dt       = -v / tau_h + sigma_noise * xi                       : 1
dvth/dt     = -(vth - {vth_rest}) / tau_vth                       : 1
dtrace_r/dt = -trace_r / tau_r                                    : 1
"""
G_h = NeuronGroup(
    N_H, eqs_h,
    threshold="v > vth",
    reset=f"v=0; vth=vth+{vth_jump}; trace_r=1;",
    method="euler"
)

# ── STDP synapses: input → hidden ─────────────────────────────────────────────
stdp_model = """
w          : 1
dapre/dt   = -apre  / taupre  : 1 (event-driven)
dapost/dt  = -apost / taupost : 1 (event-driven)
dr1/dt     = -r1 / taupre      : 1 (event-driven)
dr2/dt     = -r2 / tau_x       : 1 (event-driven)
do1/dt     = -o1 / taupost     : 1 (event-driven)
do2/dt     = -o2 / tau_y       : 1 (event-driven)
wmax_syn   : 1
Apre_syn   : 1
Apost_syn  : 1
A3pre_syn  : 1
A3post_syn : 1
"""
# Pair terms kept verbatim; triplet term appended. Slow traces (r2, o2) are read
# BEFORE being incremented, so they carry the value from the *previous* spike.
#   on_pre  LTD: pair apost*(w-wmin)     + triplet A3post_syn*o1*r2*(w-wmin)
#   on_post LTP: pair apre*(wmax_syn-w)  + triplet A3pre_syn *r1*o2*(wmax_syn-w)
on_pre  = ("v_post += w * (1 - trace_r_post)\n"
           "apre += Apre_syn\n"
           "w = clip(w + apost*(w-wmin) + A3post_syn*o1*r2*(w-wmin), wmin, wmax_syn)\n"
           "r1 += 1\n"
           "r2 += 1")
on_post = ("apost += Apost_syn\n"
           "w = clip(w + apre*(wmax_syn-w) + A3pre_syn*r1*o2*(wmax_syn-w), wmin, wmax_syn)\n"
           "o1 += 1\n"
           "o2 += 1")

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
S_ih.connect(i=_src_ih, j=_tgt_ih)
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn   = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn   = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn  = Apost_matrix[src_ih, tgt_ih]
S_ih.A3pre_syn  = A3pre_matrix[src_ih, tgt_ih]
S_ih.A3post_syn = A3post_matrix[src_ih, tgt_ih]

# ── Inhibitory lateral synapse: hidden → hidden ────────────────────────────────
stdp_inh_model = """
w_inh          : 1
dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
wmax_inh_syn   : 1
Apre_inh_syn   : 1
Apost_inh_syn  : 1
"""
on_pre_inh  = (f"v_post -= w_inh * (1 - trace_r_post)\n"
               f"apre_inh += Apre_inh_syn\n"
               f"w_inh = clip(w_inh + apost_inh*(w_inh-{W_INH_MIN}), {W_INH_MIN}, wmax_inh_syn)")
on_post_inh = (f"apost_inh += Apost_inh_syn\n"
               f"w_inh = clip(w_inh + apre_inh*(wmax_inh_syn-w_inh), {W_INH_MIN}, wmax_inh_syn)")

S_hh = Synapses(G_h, G_h, model=stdp_inh_model, on_pre=on_pre_inh, on_post=on_post_inh)
S_hh.connect(i=_src_hh, j=_tgt_hh)

S_hh.wmax_inh_syn  = wmax_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apre_inh_syn  = Apre_inh_matrix[_src_hh, _tgt_hh]
S_hh.Apost_inh_syn = Apost_inh_matrix[_src_hh, _tgt_hh]
S_hh.w_inh         = W_HH_INIT[_src_hh, _tgt_hh]

# ── Periodic L1 normalisation every 100 ms + snapshot collection ──────────────
tgt_masks_ih     = [np.where(tgt_ih == j)[0] for j in range(N_H)]
tgt_masks_hh     = [np.where(_tgt_hh == j)[0] for j in range(N_H)]
wmax_syn_arr     = np.array(S_ih.wmax_syn)
wmax_inh_syn_arr = np.array(S_hh.wmax_inh_syn)

_run_state = {"epoch": 0, "snapshots": []}

@network_operation(dt=100*ms, when='end')
def normalize_weights():
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = np.array(S_ih.w[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_EXC:
            S_ih.w[idx] = np.clip(w_col * NORM_LIMIT_EXC / wsum, wmin, wmax_syn_arr[idx])
    for j in range(N_H):
        idx   = tgt_masks_hh[j]
        w_col = np.array(S_hh.w_inh[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_INH:
            S_hh.w_inh[idx] = np.clip(w_col * NORM_LIMIT_INH / wsum, W_INH_MIN, wmax_inh_syn_arr[idx])
    if _run_state["epoch"] >= SNAPSHOT_FROM_EPOCH:
        w_snap = np.zeros((N_IN, N_H))
        w_snap[src_ih, tgt_ih] = np.array(S_ih.w)
        _run_state["snapshots"].append(w_snap)

net = Network(G_in, G_h, S_ih, S_hh, normalize_weights)
G_h.vth = vth_init
net.store('init')

# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def build_entries():
    """Flatten WAV_FILES_A / WAV_FILES_B (person → session → path) into runs.

    One run per (part, person, session), each trained on its single audio file.
    Entries are ordered PART-major, then person, then session — so the part-A
    fingerprints (heatmap rows) and part-B fingerprints (heatmap cols) share the
    same speaker/session ordering that heatmap.py's block logic assumes.

    Missing paths / files are reported up front so typos fail fast.
    """
    # A and B must describe the same persons+sessions or the heatmap blocks misalign.
    def _structure(tree):
        return {p: sorted(tree[p]) for p in sorted(tree)}
    if _structure(WAV_FILES_A) != _structure(WAV_FILES_B):
        raise ValueError("WAV_FILES_A and WAV_FILES_B must have identical "
                         "persons and sessions (only the file paths may differ).")

    entries = []
    for part, tree in (('A', WAV_FILES_A), ('B', WAV_FILES_B)):
        for person_id in sorted(tree):
            for session_id in sorted(tree[person_id]):
                path = tree[person_id][session_id]
                if not path:
                    raise ValueError(
                        f"Part {part} {person_id}/{session_id}: empty path.")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Part {part} {person_id}/{session_id}: file not found: {path}")
                entries.append({
                    'person_id': person_id,
                    'record_id': session_id,
                    'part':      part,
                    'wav_files': [path],
                    'label':     f"{person_id}/{session_id}/part-{part}",
                })
    return entries

# ─────────────────────────────────────────────────────────────────────────────
# Single-run training
# ─────────────────────────────────────────────────────────────────────────────

def train_fingerprint(wav_files, label):
    """Train one SNN run and return the averaged weight fingerprint.

    Weight snapshots are collected inside normalize_weights() every 100 ms for
    epochs >= SNAPSHOT_FROM_EPOCH. The fingerprint is the mean of all snapshots.

    Returns np.ndarray of shape (N_IN, N_H), dtype float32.
    """
    t0   = time.time()
    w_ih = W_IH_INIT.copy()
    w_hh = W_HH_INIT.copy()
    _run_state["snapshots"] = []

    for epoch_idx in range(NUM_EPOCHS):
        _run_state["epoch"] = epoch_idx

        for wav_path in wav_files:
            try:
                I, T = compute_spike_input_current(
                    wav_path,
                    scale=1,
                    num_filters=96,
                    sustained_per_band=1,
                    onset_per_band=1,
                    phase_per_band=1,
                    sust_gain=0.3,
                    onset_gain=2.25,
                    phase_gain=0.45,
                    sust_spread_min=1,
                    sust_spread_max=1,
                )
            except Exception as e:
                print(f"    [skip {os.path.basename(wav_path)}: {e}]")
                continue

            net.restore('init')
            G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

            S_ih.w         = w_ih[src_ih, tgt_ih]
            S_ih.apre      = 0
            S_ih.apost     = 0
            S_ih.r1 = 0; S_ih.r2 = 0; S_ih.o1 = 0; S_ih.o2 = 0   # triplet detectors

            S_hh.w_inh     = w_hh[_src_hh, _tgt_hh]
            S_hh.apre_inh  = 0
            S_hh.apost_inh = 0

            net.run(T * DT_SIM)

            w_ih_new = np.zeros((N_IN, N_H))
            w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
            w_ih = w_ih_new

            w_hh_new = np.zeros((N_H, N_H))
            w_hh_new[_src_hh, _tgt_hh] = np.array(S_hh.w_inh)
            w_hh = w_hh_new

    snapshots = _run_state["snapshots"]
    fingerprint = (np.mean(np.stack(snapshots), axis=0).astype(np.float32)
                   if snapshots else w_ih.astype(np.float32))
    print(f"  {label}  →  {len(snapshots)} snapshot(s) averaged  ({time.time()-t0:.1f}s)")
    return fingerprint

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

t_start = time.time()

entries = build_entries()
n = len(entries)
n_persons = len(set(e['person_id'] for e in entries))
print(f"Prepared {n} run(s): {n_persons} person(s), "
      f"parts A={sum(e['part']=='A' for e in entries)} "
      f"B={sum(e['part']=='B' for e in entries)}.\n")

fingerprints  = []
person_ids    = []
record_ids    = []
parts         = []
wav_files_arr = []

for idx, e in enumerate(entries):
    print(f"[{idx+1:02d}/{n:02d}] {e['label']}")
    fp = train_fingerprint(e['wav_files'], e['label'])
    fingerprints.append(fp)
    person_ids.append(e['person_id'])
    record_ids.append(e['record_id'])
    parts.append(e['part'])
    wav_files_arr.append(e['wav_files'])

fingerprints = np.stack(fingerprints, axis=0)

np.savez(
    OUT_PATH,
    fingerprints = fingerprints,
    person_ids   = np.array(person_ids),
    record_ids   = np.array(record_ids),
    parts        = np.array(parts),
    wav_files    = np.array(wav_files_arr, dtype=object),
)

print(f"\n{'='*60}")
print(f"Saved → {os.path.relpath(OUT_PATH)}")
print(f"  fingerprints : {fingerprints.shape}  "
      f"range [{fingerprints.min():.4f}, {fingerprints.max():.4f}]")
print(f"  persons      : {sorted(set(person_ids))}")
print(f"  Total time   : {time.time() - t_start:.1f}s")
print(f"{'='*60}")
