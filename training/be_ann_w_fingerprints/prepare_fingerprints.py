"""
prepare_fingerprints.py
=======================
Trains an independent SNN run per wav file using the
tonotopic_plasticity_bound architecture.  Produces two output files:

  training/be_ann_w_fingerprints/dev_fingerprints.npz
  training/be_ann_w_fingerprints/test_fingerprints.npz

Dataset layout expected (output of prepare_dataset.py):
  datasets/vox1_10person_fingerprint/
    wav_dev/<person_id>/<record_id>/00001.wav … 00008.wav
    wav_test/<person_id>/<record_id>/00009.wav 00010.wav

Each wav file yields one fingerprint (fresh network weights per file).
Training: NUM_EPOCHS=4 per wav file.  Snapshots collected after each epoch
where epoch_idx >= COLLECT_FROM_EPOCH_IDX (0-indexed), i.e. epochs 3 & 4
(1-indexed).  The fingerprint is the mean of collected snapshots.

Output arrays (per npz):
  fingerprints  (N, 672, 672)  float32
  person_ids    (N,)           str
  record_ids    (N,)           str

Usage:
  cd <repo_root>
  python3 training/be_ann_w_fingerprints/prepare_fingerprints.py
"""

import argparse
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

_ap = argparse.ArgumentParser()
_ap.add_argument('--warmup', action='store_true',
                 help='Process only 1 fingerprint per split (pipeline smoke-test)')
args = _ap.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT  = os.path.join(REPO_ROOT, 'datasets', 'vox1_100person_fingerprint')
OUT_DEV_PATH  = os.path.join(SCRIPT_DIR, 'dev_fingerprints.npz')
OUT_TEST_PATH = os.path.join(SCRIPT_DIR, 'test_fingerprints.npz')

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
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
NUM_EPOCHS             = 4
COLLECT_FROM_EPOCH_IDX = 2   # 0-indexed: collect after epoch 2 and 3 (epochs 3&4, 1-indexed)

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
# Initialise weight matrices
# ─────────────────────────────────────────────────────────────────────────────

# Excitatory: formula-shaped, column-normalised to NORM_LIMIT_EXC
w_ih_init = np.zeros((N_IN, N_H))
w_ih_init[_src_ih, _tgt_ih] = wmax_matrix[_src_ih, _tgt_ih]
for _j in range(N_H):
    _col_mask = (_tgt_ih == _j)
    _rows = _src_ih[_col_mask]
    _wsum = w_ih_init[_rows, _j].sum()
    if _wsum > 0:
        w_ih_init[_rows, _j] *= NORM_LIMIT_EXC / _wsum

# Inhibitory: formula-shaped, column-normalised to NORM_LIMIT_INH
w_hh_init = np.zeros((N_H, N_H))
w_hh_init[_src_hh, _tgt_hh] = wmax_inh_matrix[_src_hh, _tgt_hh]
for _j in range(N_H):
    _col_mask = (_tgt_hh == _j)
    _rows = _src_hh[_col_mask]
    _wsum = w_hh_init[_rows, _j].sum()
    if _wsum > 0:
        w_hh_init[_rows, _j] *= NORM_LIMIT_INH / _wsum

init_weights = {"in->hid": w_ih_init.copy(), "hid->hid": w_hh_init.copy()}

# ─────────────────────────────────────────────────────────────────────────────
# Build Brian2 network (once — reused across all runs via store/restore)
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
S_hh.w_inh         = w_hh_init[_src_hh, _tgt_hh]

# ── Periodic L1 normalisation every 500 ms — excitatory and inhibitory ────────
tgt_masks_ih     = [np.where(tgt_ih == j)[0] for j in range(N_H)]
tgt_masks_hh     = [np.where(_tgt_hh == j)[0] for j in range(N_H)]
wmax_syn_arr     = np.array(S_ih.wmax_syn)
wmax_inh_syn_arr = np.array(S_hh.wmax_inh_syn)

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

net = Network(G_in, G_h, S_ih, S_hh, normalize_weights)
G_h.vth = vth_init
net.store('init')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _final_normalize(w_flat, w_inh_flat):
    """One L1 normalisation pass on both excitatory and inhibitory weights."""
    for j in range(N_H):
        idx   = tgt_masks_ih[j]
        w_col = w_flat[idx]
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_EXC:
            w_flat[idx] = np.clip(w_col * NORM_LIMIT_EXC / wsum, wmin, wmax_syn_arr[idx])
    for j in range(N_H):
        idx   = tgt_masks_hh[j]
        w_col = w_inh_flat[idx]
        wsum  = w_col.sum()
        if wsum > NORM_LIMIT_INH:
            w_inh_flat[idx] = np.clip(w_col * NORM_LIMIT_INH / wsum, W_INH_MIN, wmax_inh_syn_arr[idx])
    return w_flat, w_inh_flat


def discover_split(split_dir):
    """Return list of (person_id, record_id, wav_path) for every wav in split_dir."""
    entries = []
    for person_id in sorted(os.listdir(split_dir)):
        person_dir = os.path.join(split_dir, person_id)
        if not os.path.isdir(person_dir):
            continue
        for record_id in sorted(os.listdir(person_dir)):
            record_dir = os.path.join(person_dir, record_id)
            if not os.path.isdir(record_dir):
                continue
            for fname in sorted(f for f in os.listdir(record_dir) if f.endswith('.wav')):
                entries.append({
                    'person_id': person_id,
                    'record_id': record_id,
                    'wav_path':  os.path.join(record_dir, fname),
                    'label':     f"{person_id}/{record_id}/{fname}",
                })
    return entries


def train_fingerprint(wav_path, label):
    """Train fresh SNN for NUM_EPOCHS on one wav file; return averaged fingerprint.

    Weights reset to init_weights at the start. Snapshots collected from
    COLLECT_FROM_EPOCH_IDX onward (0-indexed). Returns (N_IN, N_H) float32.
    """
    t0 = time.time()

    w_ih = init_weights["in->hid"].copy()
    w_hh = init_weights["hid->hid"].copy()

    try:
        I, T = compute_spike_input_current(
            wav_path, scale=0.8,
            num_filters=96,
            sustained_per_band=4, onset_per_band=2, phase_per_band=1,
            sust_spread_min=0.7, sust_spread_max=1.3,
        )
    except Exception as e:
        print(f"  [skip {label}: {e}]")
        return None

    collected = []

    for epoch_idx in range(NUM_EPOCHS):
        net.restore('init')
        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        S_ih.w         = w_ih[src_ih, tgt_ih]
        S_ih.apre      = 0
        S_ih.apost     = 0

        S_hh.w_inh     = w_hh[_src_hh, _tgt_hh]
        S_hh.apre_inh  = 0
        S_hh.apost_inh = 0

        net.run(T * DT_SIM)

        w_raw     = np.array(S_ih.w)
        w_inh_raw = np.array(S_hh.w_inh)
        w_raw, w_inh_raw = _final_normalize(w_raw, w_inh_raw)
        S_ih.w    = w_raw
        S_hh.w_inh = w_inh_raw

        w_ih_new = np.zeros((N_IN, N_H))
        w_ih_new[src_ih, tgt_ih] = w_raw
        w_ih = w_ih_new

        w_hh_new = np.zeros((N_H, N_H))
        w_hh_new[_src_hh, _tgt_hh] = w_inh_raw
        w_hh = w_hh_new

        if epoch_idx >= COLLECT_FROM_EPOCH_IDX:
            collected.append(w_ih.copy())

    fingerprint = (np.mean(collected, axis=0).astype(np.float32)
                   if collected else w_ih.astype(np.float32))
    print(f"  {label}  →  {len(collected)} snapshot(s)  ({time.time()-t0:.1f}s)")
    return fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

t_start = time.time()

for split_name, out_path in (('wav_dev', OUT_DEV_PATH), ('wav_test', OUT_TEST_PATH)):
    split_dir = os.path.join(DATASET_ROOT, split_name)
    entries   = discover_split(split_dir)
    if args.warmup:
        entries = entries[:1]
    n         = len(entries)
    persons   = sorted(set(e['person_id'] for e in entries))
    print(f"\n{'='*60}")
    print(f"Split: {split_name}  |  {n} wav file(s) across {len(persons)} person(s)")
    print(f"{'='*60}")

    fingerprints = []
    person_ids   = []
    record_ids   = []

    for idx, e in enumerate(entries):
        print(f"[{idx+1:03d}/{n:03d}] {e['label']}")
        fp = train_fingerprint(e['wav_path'], e['label'])
        if fp is None:
            continue
        fingerprints.append(fp)
        person_ids.append(e['person_id'])
        record_ids.append(e['record_id'])

    fingerprints_arr = np.stack(fingerprints, axis=0)

    np.savez(
        out_path,
        fingerprints = fingerprints_arr,
        person_ids   = np.array(person_ids),
        record_ids   = np.array(record_ids),
    )

    print(f"\nSaved → {os.path.relpath(out_path)}")
    print(f"  fingerprints : {fingerprints_arr.shape}  "
          f"range [{fingerprints_arr.min():.4f}, {fingerprints_arr.max():.4f}]")
    print(f"  persons      : {persons}")

print(f"\nTotal time: {time.time() - t_start:.1f}s")
