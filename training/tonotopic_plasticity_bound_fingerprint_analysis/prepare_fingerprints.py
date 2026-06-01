"""
prepare_fingerprints.py
=======================
Trains independent SNN runs (one per speaker/recording/part triplet) using the
tonotopic_plasticity_bound architecture and saves weight fingerprints.

Dataset layout expected:
  datasets/vox1_fingerprint_analysis/
    <person_id>/
      <record_id>/
        00001.wav  00002.wav  00003.wav  00004.wav

Each (speaker, recording) pair yields two fingerprints:
  Part A — trained on samples 00001.wav + 00002.wav
  Part B — trained on samples 00003.wav + 00004.wav

Training: NUM_EPOCHS per run.  Weight snapshots are collected inside
normalize_weights() every 500 ms for epochs >= SNAPSHOT_FROM_EPOCH.
The fingerprint is the mean of all collected snapshots (float32).

All runs share the same initial weight matrices (seed 42) so differences between
fingerprints reflect audio content, not random initialisation.

Output: fingerprints.npz
  fingerprints  (N, 672, 672)  float32  — averaged weight matrices
  person_ids    (N,)           str
  record_ids    (N,)           str
  parts         (N,)           str      — "A" or "B"
  wav_files     (N, 2)         str

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

DATASET_ROOT = os.path.join(REPO_ROOT, 'datasets', 'vox1_fingerprint_analysis')
OUT_PATH     = os.path.join(SCRIPT_DIR, 'fingerprints.npz')

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (match training/tonotopic_plasticity_bound/train.py)
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 672   # 96 channels × 7 neurons/channel
N_H  = 672

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

# -- Excitatory weight bounds --
wmin = 0.0

# -- Excitatory synapse --
WMAX_CENTER  = 1.0
APRE_CENTER  =  0.002
APOST_CENTER = -0.0024

# -- Inhibitory lateral synapse --
W_INH_CENTER = 1.0
W_INH_MIN    = 0.0
APRE_INH     = 0.0005
APOST_INH    = -0.0006

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d/R)^p)) --
R_EXC = 128
p_EXC = 3
R_INH = 7
p_INH = 10000

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 2
NORM_LIMIT_INH = 0.9

# -- Fingerprint collection --
NUM_EPOCHS         = 2
SNAPSHOT_FROM_EPOCH = 1   # collect snapshots from this epoch onward (0-indexed)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity matrices — excitatory
# topo(d) = max(0, 1-(d/R)^p); hard cutoff at d=R_EXC.
# ─────────────────────────────────────────────────────────────────────────────

_i       = np.arange(N_IN).reshape(-1, 1)
_j       = np.arange(N_H).reshape(1, -1)
_dist_ih = np.abs(_i - _j)
_dist_ih = np.minimum(_dist_ih, N_IN - _dist_ih)
_topo_exc = np.maximum(0.0, 1.0 - (_dist_ih / R_EXC) ** p_EXC)
_mask_ih  = _dist_ih <= R_EXC
_src_ih, _tgt_ih = np.where(_mask_ih)

wmax_matrix  = WMAX_CENTER  * _topo_exc
Apre_matrix  = APRE_CENTER  * _topo_exc
Apost_matrix = APOST_CENTER * _topo_exc
del _i, _j, _dist_ih, _topo_exc

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity matrices — inhibitory
# Same formula; no self-connections (d > 0); hard cutoff at d=R_INH.
# ─────────────────────────────────────────────────────────────────────────────

_i_hh    = np.arange(N_H).reshape(-1, 1)
_j_hh    = np.arange(N_H).reshape(1, -1)
_dist_hh = np.abs(_i_hh - _j_hh)
_dist_hh = np.minimum(_dist_hh, N_H - _dist_hh)
_topo_inh = np.maximum(0.0, 1.0 - (_dist_hh / R_INH) ** p_INH)
_mask_hh  = (_dist_hh > 0) & (_dist_hh <= R_INH)
_src_hh, _tgt_hh = np.where(_mask_hh)

wmax_inh_matrix  = W_INH_CENTER * _topo_inh
Apre_inh_matrix  = APRE_INH     * _topo_inh
Apost_inh_matrix = APOST_INH    * _topo_inh
del _i_hh, _j_hh, _dist_hh, _topo_inh

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrices — same for all runs
# Excitatory: formula-shaped, column-normalised to NORM_LIMIT_EXC
# Inhibitory: formula-shaped, column-normalised to NORM_LIMIT_INH
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
W_HH_INIT[_src_hh, _tgt_hh] = wmax_inh_matrix[_src_hh, _tgt_hh]
for _j in range(N_H):
    _col_mask = (_tgt_hh == _j)
    _rows = _src_hh[_col_mask]
    _wsum = W_HH_INIT[_rows, _j].sum()
    if _wsum > 0:
        W_HH_INIT[_rows, _j] *= NORM_LIMIT_INH / _wsum

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
wmax_syn   : 1
Apre_syn   : 1
Apost_syn  : 1
"""
on_pre  = "v_post += w * (1 - trace_r_post)\napre += Apre_syn\nw = clip(w + apost*(w-wmin), wmin, wmax_syn)"
on_post = "apost += Apost_syn\nw = clip(w + apre*(wmax_syn-w), wmin, wmax_syn)"

S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
S_ih.connect(i=_src_ih, j=_tgt_ih)
src_ih = np.array(S_ih.i)
tgt_ih = np.array(S_ih.j)

S_ih.wmax_syn  = wmax_matrix[src_ih, tgt_ih]
S_ih.Apre_syn  = Apre_matrix[src_ih, tgt_ih]
S_ih.Apost_syn = Apost_matrix[src_ih, tgt_ih]

# ── Inhibitory lateral synapse: hidden → hidden ────────────────────────────────
stdp_inh_model = """
w_inh          : 1
dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
wmax_inh_syn   : 1
Apre_inh_syn   : 1
Apost_inh_syn  : 1
"""
on_pre_inh  = (f"v_post -= w_inh\n"
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

# ── Periodic L1 normalisation every 500 ms + snapshot collection ──────────────
tgt_masks_ih     = [np.where(tgt_ih == j)[0] for j in range(N_H)]
tgt_masks_hh     = [np.where(_tgt_hh == j)[0] for j in range(N_H)]
wmax_syn_arr     = np.array(S_ih.wmax_syn)
wmax_inh_syn_arr = np.array(S_hh.wmax_inh_syn)

_run_state = {"epoch": 0, "snapshots": []}

@network_operation(dt=500*ms, when='end')
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

def discover_dataset(root):
    entries = []
    for person_id in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person_id)
        if not os.path.isdir(person_dir):
            continue
        for record_id in sorted(os.listdir(person_dir)):
            record_dir = os.path.join(person_dir, record_id)
            if not os.path.isdir(record_dir):
                continue
            for part, filenames in (('A', ['00001.wav', '00002.wav']),
                                    ('B', ['00003.wav', '00004.wav'])):
                wav_paths = [os.path.join(record_dir, f) for f in filenames]
                missing = [p for p in wav_paths if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(f"Missing expected files: {missing}")
                entries.append({
                    'person_id': person_id,
                    'record_id': record_id,
                    'part':      part,
                    'wav_files': wav_paths,
                    'label':     f"{person_id}/{record_id}/part-{part}",
                })
    return entries

# ─────────────────────────────────────────────────────────────────────────────
# Single-run training
# ─────────────────────────────────────────────────────────────────────────────

def train_fingerprint(wav_files, label):
    """Train one SNN run and return the averaged weight fingerprint.

    Weight snapshots are collected inside normalize_weights() every 500 ms for
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
                    wav_path, scale=0.8,
                    num_filters=96,
                    sustained_per_band=4, onset_per_band=2, phase_per_band=1,
                    sust_spread_min=0.7, sust_spread_max=1.3,
                )
            except Exception as e:
                print(f"    [skip {os.path.basename(wav_path)}: {e}]")
                continue

            net.restore('init')
            G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

            S_ih.w         = w_ih[src_ih, tgt_ih]
            S_ih.apre      = 0
            S_ih.apost     = 0

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

entries = discover_dataset(DATASET_ROOT)
n = len(entries)
print(f"Discovered {n} run(s) across "
      f"{len(set(e['person_id'] for e in entries))} person(s).\n")

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
    wav_files    = np.array(wav_files_arr),
)

print(f"\n{'='*60}")
print(f"Saved → {os.path.relpath(OUT_PATH)}")
print(f"  fingerprints : {fingerprints.shape}  "
      f"range [{fingerprints.min():.4f}, {fingerprints.max():.4f}]")
print(f"  persons      : {sorted(set(person_ids))}")
print(f"  Total time   : {time.time() - t_start:.1f}s")
print(f"{'='*60}")
