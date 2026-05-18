"""
prepare_fingerprints.py
=======================
Trains independent SNN runs (one per speaker/recording/part triplet) using the
full tonotopic_plasticity_bound architecture and saves weight fingerprints.

Dataset layout expected:
  datasets/vox1_fingerprint_analysis/
    <person_id>/
      <record_id>/
        00001.wav  00002.wav  00003.wav  00004.wav

Each (speaker, recording) pair yields two fingerprints:
  Part A — trained on samples 00001.wav + 00002.wav
  Part B — trained on samples 00003.wav + 00004.wav

Training: NUM_EPOCHS per run.  Weight snapshots are collected after every sample
in epoch >= COLLECT_FROM_EPOCH.  Before each collection the weights are
explicitly L1-normalised (in addition to the 500ms periodic norm that runs
during the simulation) so every snapshot is guaranteed to be post-normalisation.
The fingerprint is the mean of all collected snapshots.

All runs share the same initial weight matrix (seed 42) so differences between
fingerprints reflect audio content, not random initialisation.

Output: fingerprints.npz
  fingerprints  (N, 700, 700)  float32  — averaged weight matrices
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
from src.utils.weights_utils import gaussian_weight_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = os.path.join(REPO_ROOT, 'datasets', 'vox1_fingerprint_analysis')
OUT_PATH     = os.path.join(SCRIPT_DIR, 'fingerprints.npz')

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (match training/tonotopic_plasticity_bound/train.py)
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 700
N_H  = 700

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

# -- Tonotopic plasticity bounds — excitatory (circular Gaussian) --
WMAX_CENTER  = 1.0
WMAX_MIN     = 0.3
APRE_CENTER  =  0.002
APOST_CENTER = -0.0024
PLAST_SIGMA  = N_IN / 5

# -- Inhibitory lateral synapse (distance-limited STDP) --
SIGMA_INH       = N_H / 5
W_INH_CENTER    = 0.3
W_INH_MIN       = 0.0
APRE_INH        = 0.002
APOST_INH       = -0.0024
INH_CUTOFF_MULT = 3

# -- Weight initialisation --
W_INIT_SIGMA     = N_IN / 5
W_INIT_NOISE_STD = 0.005
W_INIT_SUM       = 2

# -- Homeostatic normalisation --
NORM_LIMIT = 2

# -- WTA --
WTA_K = 35

# -- Fingerprint collection --
NUM_EPOCHS         = 2
COLLECT_FROM_EPOCH = 1   # collect snapshots from this epoch onward (0-indexed)

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrices — same for all runs
# ─────────────────────────────────────────────────────────────────────────────

W_INIT = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
                                W_INIT_SUM, wmin, WMAX_CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity bound matrices — excitatory
# ─────────────────────────────────────────────────────────────────────────────

_i    = np.arange(N_IN).reshape(-1, 1)
_j    = np.arange(N_H).reshape(1, -1)
_dist = np.abs(_i - _j)
_dist = np.minimum(_dist, N_IN - _dist)
_gauss = np.exp(-(_dist ** 2) / (2 * PLAST_SIGMA ** 2))

wmax_matrix  = WMAX_MIN + (WMAX_CENTER - WMAX_MIN) * _gauss   # [0.3, 1.0]
Apre_matrix  = APRE_CENTER  * _gauss
Apost_matrix = APOST_CENTER * _gauss
del _i, _j, _dist, _gauss

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute tonotopic plasticity bound matrices — inhibitory
# ─────────────────────────────────────────────────────────────────────────────

_i_hh    = np.arange(N_H).reshape(-1, 1)
_j_hh    = np.arange(N_H).reshape(1, -1)
_dist_hh = np.abs(_i_hh - _j_hh)
_dist_hh = np.minimum(_dist_hh, N_H - _dist_hh)
_gauss_hh = np.exp(-(_dist_hh ** 2) / (2 * SIGMA_INH ** 2))

wmax_inh_matrix  = W_INH_CENTER * _gauss_hh
Apre_inh_matrix  = APRE_INH     * _gauss_hh
Apost_inh_matrix = APOST_INH    * _gauss_hh

_cutoff_hh       = INH_CUTOFF_MULT * SIGMA_INH
_mask_hh         = (_dist_hh > 0) & (_dist_hh <= _cutoff_hh)
_src_hh, _tgt_hh = np.where(_mask_hh)
del _i_hh, _j_hh, _dist_hh, _gauss_hh

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
is_winner                                                         : boolean
"""
G_h = NeuronGroup(
    N_H, eqs_h,
    threshold="v > vth and is_winner",
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
S_ih.connect()
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
S_hh.w_inh         = 0.0

# ── WTA ───────────────────────────────────────────────────────────────────────
@network_operation(when="before_thresholds")
def determine_winner():
    v   = G_h.v[:]
    vth = G_h.vth[:]
    crossed = v > vth
    if np.any(crossed):
        candidates = np.where(crossed)[0]
        sorted_idx = candidates[np.argsort(vth[candidates])]
        winners    = sorted_idx[:WTA_K]
        G_h.is_winner[:]       = False
        G_h.is_winner[winners] = True
        G_h.v[np.setdiff1d(candidates, winners)] = 0.5
    else:
        G_h.is_winner[:] = False

# ── Periodic L1 normalisation every 500 ms ────────────────────────────────────
tgt_masks_ih = [np.where(tgt_ih == j)[0] for j in range(N_H)]
wmax_syn_arr  = np.array(S_ih.wmax_syn)   # static per-synapse wmax, cache once

@network_operation(dt=500*ms, when='end')
def normalize_weights():
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = np.array(S_ih.w[idx])
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT and NORM_LIMIT > 0:
            S_ih.w[idx] = np.clip(w_col * NORM_LIMIT / wsum, wmin, wmax_syn_arr[idx])

net = Network(G_in, G_h, S_ih, S_hh, determine_winner, normalize_weights)
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

def _final_normalize(w_flat):
    """Apply one L1 normalization pass to a flat synapse weight array in-place.

    The 500ms periodic op fires at fixed ticks and may not align with the end of
    the simulation. This guarantees every collected snapshot is post-normalization.
    """
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = w_flat[idx]
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT and NORM_LIMIT > 0:
            w_flat[idx] = np.clip(w_col * NORM_LIMIT / wsum, wmin, wmax_syn_arr[idx])
    return w_flat


def train_fingerprint(wav_files, label):
    """Train one SNN run and return the averaged weight fingerprint.

    Weight snapshots are collected after every sample in epochs >=
    COLLECT_FROM_EPOCH. Each snapshot is explicitly normalized before
    collection. The fingerprint is the mean of all snapshots (float32).

    Returns np.ndarray of shape (N_IN, N_H), dtype float32.
    """
    t0   = time.time()
    w_ih = W_INIT.copy()
    w_hh = np.zeros((N_H, N_H))

    collected = []

    for epoch_idx in range(NUM_EPOCHS):
        for wav_path in wav_files:
            try:
                I, T = compute_spike_input_current(
                    wav_path, scale=0.8,
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

            # Final normalization pass — guarantees snapshot is post-normalization
            w_raw = np.array(S_ih.w)
            w_raw = _final_normalize(w_raw)
            S_ih.w = w_raw   # write back so next sample inherits normalized weights

            # Extract weight matrices (w_hh carries over to next sample)
            w_ih_new = np.zeros((N_IN, N_H))
            w_ih_new[src_ih, tgt_ih] = w_raw
            w_ih = w_ih_new

            w_hh_new = np.zeros((N_H, N_H))
            w_hh_new[_src_hh, _tgt_hh] = np.array(S_hh.w_inh)
            w_hh = w_hh_new

            if epoch_idx >= COLLECT_FROM_EPOCH:
                collected.append(w_ih.copy())

    fingerprint = (np.mean(collected, axis=0).astype(np.float32)
                   if collected else w_ih.astype(np.float32))
    print(f"  {label}  →  {len(collected)} snapshot(s) averaged  ({time.time()-t0:.1f}s)")
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
