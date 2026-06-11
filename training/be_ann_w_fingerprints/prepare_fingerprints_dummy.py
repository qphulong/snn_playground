"""
prepare_fingerprints_dummy.py
=============================
Single-file smoke test for the new fingerprint architecture (channel-distance
tonotopic excitation + Jaccard-overlap lateral inhibition).  Trains one SNN run
on ONE hardcoded wav file and saves ONE fingerprint, already transformed into the
ANN-ready (2, 7, 33, 672) tensor.

Input wav:
  datasets/vox1_cleaned/wav_dev/id10010/id10010_00003/00001.wav

Training: NUM_EPOCHS per run.  Weight snapshots are collected inside
normalize_weights() every 500 ms for epochs >= SNAPSHOT_FROM_EPOCH.
The averaged (672, 672) weight matrix is then unrolled into the tensor below.

Tensor transform (the (672, 672) excitatory matrix → (2, 7, 33, 672)):
  axis 0 (2)   norm channel: 0 = raw weight, 1 = z-scored per hidden column
  axis 1 (7)   within-channel input-neuron type (4 sustained, 2 onset, 1 phase)
  axis 2 (33)  channel offset -16..+16 around each hidden neuron's channel
               (full connectivity window |d_ch| <= R_EXC_CHANNEL; offsets +-16
                carry weight 0 since topo = max(0, 1-(d/16)^3) is 0 at d=16)
  axis 3 (672) hidden neuron j; ch_j = j // 7, in_idx = ((ch_j+o) % 96)*7 + t

Initial weight matrices are seeded (seed 42) so the run is reproducible.

Output: dummy_fingerprint.npz
  fingerprint  (2, 7, 33, 672)  float32  — ANN-ready tensor (raw + z-scored)
  person_id    ()              str
  record_id    ()              str
  label        ()              str

Usage:
  cd <repo_root>
  python training/be_ann_w_fingerprints/prepare_fingerprints_dummy.py
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

WAV_PATH = os.path.join(
    REPO_ROOT, 'datasets', 'vox1_cleaned', 'wav_dev',
    'id10010', 'id10010_00003', '00001.wav',
)
OUT_PATH = os.path.join(SCRIPT_DIR, 'dummy_fingerprint.npz')

PERSON_ID = 'id10010'
RECORD_ID = 'id10010_00003'
LABEL     = f'{PERSON_ID}/{RECORD_ID}/00001.wav'

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

# -- Channel layout --
N_CHANNELS    = 96
N_PER_CHANNEL = N_IN // N_CHANNELS   # 7

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d_channel/R)^p)) --
R_EXC_CHANNEL = 16
p_EXC         = 3

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 2
NORM_LIMIT_INH = 0.9

# -- Fingerprint collection --
NUM_EPOCHS         = 2
SNAPSHOT_FROM_EPOCH = 1   # collect snapshots from this epoch onward (0-indexed)

# -- Tensor transform (fingerprint (672,672) -> (2, 7, 33, 672)) --
OFFSETS = np.arange(-R_EXC_CHANNEL, R_EXC_CHANNEL + 1)   # -16..+16  (33 values)

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

wmax_matrix  = WMAX_CENTER  * _topo_exc
Apre_matrix  = APRE_CENTER  * _topo_exc
Apost_matrix = APOST_CENTER * _topo_exc
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
# Single-run training
# ─────────────────────────────────────────────────────────────────────────────

def train_fingerprint(wav_path, label):
    """Train one SNN run on a single wav and return the averaged weight fingerprint.

    Weight snapshots are collected inside normalize_weights() every 500 ms for
    epochs >= SNAPSHOT_FROM_EPOCH. The fingerprint is the mean of all snapshots.

    Returns np.ndarray of shape (N_IN, N_H), dtype float32.
    """
    t0   = time.time()
    w_ih = W_IH_INIT.copy()
    w_hh = W_HH_INIT.copy()
    _run_state["snapshots"] = []

    I, T = compute_spike_input_current(
        wav_path,
        scale=1.0,
        num_filters=96,
        sustained_per_band=4,
        onset_per_band=2,
        phase_per_band=1,
        sust_gain=0.3,
        onset_gain=2.25,
        phase_gain=0.45,
        sust_spread_min=0.7,
        sust_spread_max=1.0,
    )

    for epoch_idx in range(NUM_EPOCHS):
        _run_state["epoch"] = epoch_idx

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
# Tensor transform: (672, 672) excitatory matrix → (2, 7, 33, 672)
# ─────────────────────────────────────────────────────────────────────────────

def _build_type_image(fp, t):
    """Unroll the tonotopic band for input-neuron type t → (33, 672) float32."""
    j      = np.arange(N_H)
    ch_j   = j // N_PER_CHANNEL                                   # (672,)
    in_ch  = (ch_j[None, :] + OFFSETS[:, None]) % N_CHANNELS      # (33, 672)
    in_idx = in_ch * N_PER_CHANNEL + t                           # (33, 672)
    return fp[in_idx, j[None, :]].astype(np.float32)             # (33, 672)


def _zscore_cols(img):
    """Z-score each hidden column down the 33 offset rows → zero-mean / unit-std."""
    mu  = img.mean(axis=0, keepdims=True)
    std = img.std(axis=0, keepdims=True)
    return (img - mu) / (std + 1e-8)


def fingerprint_to_tensor(fp):
    """(672, 672) → (2, 7, 33, 672): [raw, zscore] x 7 types x 33 offsets x 672 hidden."""
    raw = np.stack([_build_type_image(fp, t) for t in range(N_PER_CHANNEL)])   # (7,33,672)
    z   = np.stack([_zscore_cols(raw[t])     for t in range(N_PER_CHANNEL)])   # (7,33,672)
    return np.stack([raw, z]).astype(np.float32)                              # (2,7,33,672)

# ─────────────────────────────────────────────────────────────────────────────
# Main — single file
# ─────────────────────────────────────────────────────────────────────────────

t_start = time.time()

print(f"Processing 1 file: {LABEL}")
fp     = train_fingerprint(WAV_PATH, LABEL)
tensor = fingerprint_to_tensor(fp)                  # (2, 7, 33, 672)

np.savez(
    OUT_PATH,
    fingerprint = tensor,
    person_id   = np.array(PERSON_ID),
    record_id   = np.array(RECORD_ID),
    label       = np.array(LABEL),
)

print(f"\n{'='*60}")
print(f"Saved → {os.path.relpath(OUT_PATH)}")
print(f"  fingerprint : {tensor.shape}  (raw range [{tensor[0].min():.4f}, "
      f"{tensor[0].max():.4f}])")
print(f"  person      : {PERSON_ID}  record: {RECORD_ID}")
print(f"  Total time  : {time.time() - t_start:.1f}s")
print(f"{'='*60}")
