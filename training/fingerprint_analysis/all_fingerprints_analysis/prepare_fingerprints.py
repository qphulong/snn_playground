"""
prepare_fingerprints.py
=======================
**Run this script first** before any other analysis script in this directory.

Trains 30 independent SNN runs (5 speakers × 3 recordings × 2 halves) and
saves the resulting weight fingerprints to fingerprints.npz.

Dataset layout expected:
  datasets/vox1_fingerprint_analysis/
    <person_id>/
      <record_id>/
        00001.wav  00002.wav  00003.wav  00004.wav

Each (speaker, recording) pair yields two fingerprints:
  Part A — trained on samples 00001.wav + 00002.wav
  Part B — trained on samples 00003.wav + 00004.wav

Training: 2 epochs per run.  Weight matrices are collected after every sample
in epoch 1 (0-indexed) and averaged to produce the fingerprint.

All 30 runs share the same initial weight matrix (seed 42) so differences
between fingerprints reflect audio content, not random initialisation.

Output: fingerprints.npz
  fingerprints  (30, 700, 700)  float32  — averaged weight matrices
  person_ids    (30,)           str      — e.g. "id10545"
  record_ids    (30,)           str      — e.g. "id10545_00007"
  parts         (30,)           str      — "A" or "B"
  wav_files     (30, 2)         str      — the two WAV paths used per run

Index order: persons (sorted) → records (sorted) → [A, B]
  index 0 = person[0], record[0], part A
  index 1 = person[0], record[0], part B
  index 2 = person[0], record[1], part A
  ...

Usage:
  cd <repo_root>
  python training/fingerprint_analysis/prepare_fingerprints.py
"""

import os
import sys
import time

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network, network_operation,
    start_scope, defaultclock, ms,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.utils.weights_utils import gaussian_weight_matrix, l1_normalise_weights

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.join(SCRIPT_DIR, '..', '..', '..')
DATASET_ROOT = os.path.join(REPO_ROOT, 'datasets', 'vox1_fingerprint_analysis')
OUT_PATH     = os.path.join(SCRIPT_DIR, 'fingerprints.npz')

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────

N_IN = 700
N_H  = 700

DT_SIM = 1 * ms

tau_m       = 40  * ms
tau_a       = 20  * ms
tau_current = 1   * ms
beta        = 0.25
v_th_in     = 1.0

tau_h    = 50  * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

taupre      = 20  * ms
taupost     = 20  * ms
Apre_delta  =  0.004
Apost_delta = -0.0048

wmax = 1.0
wmin = 0.0

W_INIT_SIGMA     = N_IN / 5
W_INIT_NOISE_STD = 0.005
W_INIT_SUM       = 2

NORM_LIMIT = 2

NUM_EPOCHS         = 2
COLLECT_FROM_EPOCH = 1  # 0-indexed: collect weights from the second epoch onward

WTA_K = 35  # number of winners per timestep

# ─────────────────────────────────────────────────────────────────────────────
# Shared initial weight matrix — same for all 30 runs
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
W_INIT = gaussian_weight_matrix(N_IN, N_H, W_INIT_SIGMA, W_INIT_NOISE_STD,
                                W_INIT_SUM, wmin, wmax)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_dataset(root):
    """Return sorted list of run descriptors, one per (person, record, part).

    Each descriptor is a dict:
      person_id  str   e.g. "id10545"
      record_id  str   e.g. "id10545_00007"
      part       str   "A" or "B"
      wav_files  list  [path_sample1, path_sample2]
      label      str   human-readable run label
    """
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
                wav_files = [os.path.join(record_dir, f) for f in filenames]
                missing = [p for p in wav_files if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(
                        f"Missing expected files: {missing}")
                entries.append({
                    'person_id': person_id,
                    'record_id': record_id,
                    'part':      part,
                    'wav_files': wav_files,
                    'label':     f"{person_id}/{record_id}/part-{part}",
                })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Single-run training
# ─────────────────────────────────────────────────────────────────────────────

def train_fingerprint(wav_files, label):
    """Train one SNN run and return the averaged weight fingerprint.

    Builds a fresh Brian2 network each call.  Weight matrices are collected
    after every sample in epochs >= COLLECT_FROM_EPOCH and averaged.

    Returns np.ndarray of shape (N_IN, N_H), dtype float32.
    """
    t0   = time.time()
    w_ih = W_INIT.copy()

    start_scope()
    defaultclock.dt = DT_SIM

    _dummy_I = np.zeros((1, N_IN), dtype=float)
    I_timed  = TimedArray(_dummy_I, dt=DT_SIM)

    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in,
                       threshold="v > v_th_in",
                       reset="v=0; a+=beta",
                       refractory=2 * ms, method="euler")

    eqs_h = f"""
    dv/dt   = -v / tau_h                    : 1
    dvth/dt = -(vth - {vth_rest}) / tau_vth : 1
    is_winner                               : boolean
    """
    G_h = NeuronGroup(N_H, eqs_h,
                      threshold="v > vth and is_winner",
                      reset=f"v=0; vth=vth+{vth_jump};",
                      refractory=2 * ms, method="euler")

    stdp_model = """
    w         : 1
    dapre/dt  = -apre  / taupre  : 1 (event-driven)
    dapost/dt = -apost / taupost : 1 (event-driven)
    """
    on_pre  = "v_post += w\napre += Apre_delta\nw = clip(w + apost*(w-wmin), wmin, wmax)"
    on_post = "apost += Apost_delta\nw = clip(w + apre*(wmax-w), wmin, wmax)"

    S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
    S_ih.connect()
    src_ih = np.array(S_ih.i)
    tgt_ih = np.array(S_ih.j)

    lat = Synapses(G_h, G_h, on_pre="v_post = clip(v_post, 0, inf)")
    lat.connect(condition="i != j")

    @network_operation(when="before_thresholds")
    def determine_winner():
        v       = G_h.v[:]
        vth_arr = G_h.vth[:]
        crossed = v > vth_arr
        if np.any(crossed):
            candidates = np.where(crossed)[0]
            sorted_idx = candidates[np.argsort(vth_arr[candidates])]
            winners    = sorted_idx[:WTA_K]
            G_h.is_winner[:]       = False
            G_h.is_winner[winners] = True
            G_h.v[np.setdiff1d(candidates, winners)] = 0.5
        else:
            G_h.is_winner[:] = False

    spike_mon_h = SpikeMonitor(G_h)

    net = Network(G_in, G_h, S_ih, lat, determine_winner, spike_mon_h)
    G_h.vth = vth_init
    net.store('init')

    collected = []

    for epoch_idx in range(NUM_EPOCHS):
        for sample_idx, wav_path in enumerate(wav_files):
            try:
                I, T = compute_spike_input_current(
                    wav_path, scale=1,
                    sustained_per_band=4, onset_per_band=2, phase_per_band=1,
                    sust_spread_min=0.7, sust_spread_max=1.3,
                )
            except Exception as e:
                print(f"    [skip {os.path.basename(wav_path)}: {e}]")
                continue

            net.restore('init')
            G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)
            S_ih.w    = w_ih[src_ih, tgt_ih]
            S_ih.apre = 0
            S_ih.apost = 0

            net.run(T * DT_SIM)

            w_ih_new = np.zeros((N_IN, N_H))
            w_ih_new[src_ih, tgt_ih] = np.array(S_ih.w)
            spiked = np.unique(np.array(spike_mon_h.i))
            w_ih = l1_normalise_weights(w_ih_new, spiked, NORM_LIMIT, wmin, wmax)

            if epoch_idx >= COLLECT_FROM_EPOCH:
                collected.append(w_ih.copy())

    fingerprint = (np.mean(collected, axis=0).astype(np.float32)
                   if collected else w_ih.astype(np.float32))
    print(f"  {label}  →  {len(collected)} matrices averaged  ({time.time()-t0:.1f}s)")
    return fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

t_start = time.time()

entries = discover_dataset(DATASET_ROOT)
assert len(entries) == 30, (
    f"Expected 30 runs (5 persons × 3 records × 2 halves), found {len(entries)}"
)

print(f"Discovered {len(entries)} runs across "
      f"{len(set(e['person_id'] for e in entries))} persons.\n")

fingerprints  = []
person_ids    = []
record_ids    = []
parts         = []
wav_files_arr = []

for idx, e in enumerate(entries):
    print(f"[{idx+1:02d}/30] {e['label']}")
    fp = train_fingerprint(e['wav_files'], e['label'])
    fingerprints.append(fp)
    person_ids.append(e['person_id'])
    record_ids.append(e['record_id'])
    parts.append(e['part'])
    wav_files_arr.append(e['wav_files'])

fingerprints = np.stack(fingerprints, axis=0)  # (30, 700, 700)

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
print(f"  parts        : {list(parts[:6])} ...")
print(f"  Total time   : {time.time() - t_start:.1f}s")
print(f"{'='*60}")
