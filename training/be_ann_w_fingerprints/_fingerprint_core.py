"""
_fingerprint_core.py
====================
Shared core for the fingerprint generators (prepare_fingerprints.py and
prepare_fingerprints_dummy.py). Single source of truth for the **reduced 4-neuron
architecture** (mirrors training/tonotopic_plasticity_bound/train.py) plus the
silence-masking output format.

A fresh Brian2 network is built per process via build_network() — never at import,
because already-built Brian2 objects are not safe to fork and run concurrently. Only
the read-only precomputed matrices/index-maps live at module level.

Per-wav pipeline:
  train_fingerprint() : 4-epoch SNN run, snapshots collected in the final epoch
    (SNAPSHOT_FROM_EPOCH=3). Returns the mean excitatory weight matrix (384, 384)
    plus per-neuron spike counts for input and hidden, measured over the final-epoch
    window (the same window the snapshots span).
  fingerprint_to_sample() : (384,384) + counts -> the ANN-ready per-sample output
    weights         (4, 33, 384)  raw type-images, silent cells zeroed
    input_activity  (384,)        per-sample [0,1] (÷99th-pctl, clip; preserves 0)
    hidden_activity (384,)        same, own denominator

Import this module BEFORE numpy in driver scripts so the BLAS thread pinning below
takes effect (prevents oversubscription when many worker processes run in parallel).
"""

import os

# ── Pin BLAS threads BEFORE numpy is imported anywhere in the process ──────────
# Each fingerprint is a single-threaded Brian2/Cython sim; we parallelise across
# wavs with multiple processes, so each process must not also spin up a BLAS pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from types import SimpleNamespace

import numpy as np

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network, network_operation,
    defaultclock, prefs, BrianLogger, ms, second,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, REPO_ROOT)
from src.utils.spike_encoding import compute_spike_input_current

# ── Brian2 codegen / logging prefs ─────────────────────────────────────────────
prefs.codegen.target = 'cython'                          # JIT to C (gcc present)
prefs.codegen.runtime.cython.multiprocess_safe = True    # safe parallel build cache
BrianLogger.suppress_name('method_choice')
# The warm-compile network built once in the driver to populate the codegen cache
# is never run (discarded), which otherwise triggers an 'unused_brian_object' notice.
BrianLogger.suppress_name('unused_brian_object')
prefs.logging.console_log_level = 'ERROR'                # quiet across many workers

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters  (verbatim from tonotopic_plasticity_bound/train.py, N=384)
# ═══════════════════════════════════════════════════════════════════════════════

N_IN = 384   # 96 channels × 4 neurons/channel
N_H  = 384

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
N_PER_CHANNEL = N_IN // N_CHANNELS   # 4

# -- Tonotopic plasticity (polynomial decay: max(0, 1-(d_channel/R)^p)) --
R_EXC_CHANNEL = 16
p_EXC         = 3

# -- Homeostatic normalisation --
NORM_LIMIT_EXC = 2
NORM_LIMIT_INH = 0.9

# -- Fingerprint collection --
NUM_EPOCHS          = 4
SNAPSHOT_FROM_EPOCH = 3   # 0-indexed: only the final epoch snapshots
NORM_DT             = 500 * ms

# -- Audio encoder kwargs (match train.py) --
ENCODER_KWARGS = dict(
    scale=1.0,
    num_filters=96,
    sustained_per_band=2,
    onset_per_band=1,
    phase_per_band=1,
    sust_gain=0.3,
    onset_gain=2.25,
    phase_gain=0.45,
    sust_spread_min=0.8,
    sust_spread_max=1.0,
)

# -- Tensor transform layout --
OFFSETS = np.arange(-R_EXC_CHANNEL, R_EXC_CHANNEL + 1)   # -16..+16 (33 values)
N_OFF   = len(OFFSETS)

# Per-sample output shapes (one fingerprint):
WEIGHTS_SHAPE  = (N_PER_CHANNEL, N_OFF, N_H)   # (4, 33, 384)
ACTIVITY_SHAPE = (N_IN,)                        # (384,)

# ═══════════════════════════════════════════════════════════════════════════════
# Read-only module-level precompute (deterministic, fork-safe)
# ═══════════════════════════════════════════════════════════════════════════════

# -- Tonotopic excitatory matrices -------------------------------------------------
_ch_i    = (np.arange(N_IN) // N_PER_CHANNEL).reshape(-1, 1)
_ch_j    = (np.arange(N_H)  // N_PER_CHANNEL).reshape(1, -1)
_dist_ch = np.abs(_ch_i - _ch_j)
_dist_ch = np.minimum(_dist_ch, N_CHANNELS - _dist_ch)            # circular
_topo_exc = np.maximum(0.0, 1.0 - (_dist_ch / R_EXC_CHANNEL) ** p_EXC)
_mask_ih  = _dist_ch <= R_EXC_CHANNEL
_SRC_IH, _TGT_IH = np.where(_mask_ih)

_WMAX_MATRIX  = WMAX_CENTER  * _topo_exc
_APRE_MATRIX  = APRE_CENTER  * _topo_exc
_APOST_MATRIX = APOST_CENTER * _topo_exc
del _ch_i, _ch_j, _dist_ch, _topo_exc

# -- Jaccard inhibitory matrices ---------------------------------------------------
_ch_h       = np.arange(N_H) // N_PER_CHANNEL
_dist_ch_hh = np.abs(_ch_h.reshape(-1, 1) - _ch_h.reshape(1, -1))
_dist_ch_hh = np.minimum(_dist_ch_hh, N_CHANNELS - _dist_ch_hh)   # circular
_window_size = 2 * R_EXC_CHANNEL + 1
_overlap_ch  = np.maximum(0, _window_size - _dist_ch_hh)
_jaccard     = np.where(_overlap_ch > 0,
                        _overlap_ch / (_window_size + _dist_ch_hh), 0.0)
_mask_hh  = (_overlap_ch > 0) & (~np.eye(N_H, dtype=bool))
_SRC_HH, _TGT_HH = np.where(_mask_hh)

_WMAX_INH_MATRIX  = W_INH_CENTER * _jaccard
_APRE_INH_MATRIX  = APRE_INH     * _jaccard
_APOST_INH_MATRIX = APOST_INH    * _jaccard
del _ch_h, _dist_ch_hh, _overlap_ch, _jaccard

# -- Shared initial weight matrices (column-normalised) ----------------------------
W_IH_INIT = np.zeros((N_IN, N_H))
W_IH_INIT[_SRC_IH, _TGT_IH] = _WMAX_MATRIX[_SRC_IH, _TGT_IH]
for _j in range(N_H):
    _rows = _SRC_IH[_TGT_IH == _j]
    _wsum = W_IH_INIT[_rows, _j].sum()
    if _wsum > 0:
        W_IH_INIT[_rows, _j] *= NORM_LIMIT_EXC / _wsum

W_HH_INIT = np.zeros((N_H, N_H))
W_HH_INIT[_SRC_HH, _TGT_HH] = _WMAX_INH_MATRIX[_SRC_HH, _TGT_HH]
for _j in range(N_H):
    _rows = _SRC_HH[_TGT_HH == _j]
    _wsum = W_HH_INIT[_rows, _j].sum()
    if _wsum > 0:
        W_HH_INIT[_rows, _j] *= NORM_LIMIT_INH / _wsum

# -- Transform index map: IN_IDX[t, o, j] = ((ch_j + offset) % 96) * 4 + t ---------
# fp[IN_IDX[t], arange(N_H)] -> the (33, N_H) type-image for input-neuron type t.
_ch_j_row = (np.arange(N_H) // N_PER_CHANNEL)                       # (N_H,)
_in_ch    = (_ch_j_row[None, :] + OFFSETS[:, None]) % N_CHANNELS    # (33, N_H)
IN_IDX = np.stack([_in_ch * N_PER_CHANNEL + t for t in range(N_PER_CHANNEL)])  # (4,33,N_H)
_J_ROW = np.broadcast_to(np.arange(N_H), WEIGHTS_SHAPE)             # (4,33,N_H)
del _ch_j_row, _in_ch


# ═══════════════════════════════════════════════════════════════════════════════
# Network construction  (call once per process)
# ═══════════════════════════════════════════════════════════════════════════════

def build_network():
    """Build the Brian2 network fresh and return a handle namespace.

    Must be called once per worker process (Brian2 objects are not safe to fork
    already-built). The returned handle is passed to train_fingerprint().
    """
    defaultclock.dt = DT_SIM

    # ── Input neurons ──────────────────────────────────────────────────────────
    eqs_in = """
    dv/dt = (-v - a) / tau_m + I_timed(t, i) / tau_current : 1
    da/dt = -a / tau_a : 1
    """
    G_in = NeuronGroup(N_IN, eqs_in, threshold="v > v_th_in",
                       reset="v=0; a+=beta", refractory=2 * ms, method="euler")
    G_in.namespace["I_timed"] = TimedArray(np.zeros((1, N_IN), dtype=float), dt=DT_SIM)

    # ── Hidden neurons ─────────────────────────────────────────────────────────
    eqs_h = f"""
    dv/dt       = -v / tau_h + sigma_noise * xi                       : 1
    dvth/dt     = -(vth - {vth_rest}) / tau_vth                       : 1
    dtrace_r/dt = -trace_r / tau_r                                    : 1
    """
    G_h = NeuronGroup(N_H, eqs_h, threshold="v > vth",
                      reset=f"v=0; vth=vth+{vth_jump}; trace_r=1;", method="euler")

    # ── Excitatory STDP synapses: input → hidden ───────────────────────────────
    stdp_model = """
    w          : 1
    dapre/dt   = -apre  / taupre  : 1 (event-driven)
    dapost/dt  = -apost / taupost : 1 (event-driven)
    wmax_syn   : 1
    Apre_syn   : 1
    Apost_syn  : 1
    """
    on_pre  = ("v_post += w * (1 - trace_r_post)\napre += Apre_syn\n"
               "w = clip(w + apost*(w-wmin), wmin, wmax_syn)")
    on_post = "apost += Apost_syn\nw = clip(w + apre*(wmax_syn-w), wmin, wmax_syn)"

    S_ih = Synapses(G_in, G_h, model=stdp_model, on_pre=on_pre, on_post=on_post)
    S_ih.connect(i=_SRC_IH, j=_TGT_IH)
    src_ih = np.array(S_ih.i)
    tgt_ih = np.array(S_ih.j)
    S_ih.wmax_syn  = _WMAX_MATRIX[src_ih, tgt_ih]
    S_ih.Apre_syn  = _APRE_MATRIX[src_ih, tgt_ih]
    S_ih.Apost_syn = _APOST_MATRIX[src_ih, tgt_ih]

    # ── Inhibitory lateral STDP synapses: hidden → hidden ──────────────────────
    stdp_inh_model = """
    w_inh          : 1
    dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
    dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
    wmax_inh_syn   : 1
    Apre_inh_syn   : 1
    Apost_inh_syn  : 1
    """
    on_pre_inh  = (f"v_post -= w_inh\napre_inh += Apre_inh_syn\n"
                   f"w_inh = clip(w_inh + apost_inh*(w_inh-{W_INH_MIN}), {W_INH_MIN}, wmax_inh_syn)")
    on_post_inh = (f"apost_inh += Apost_inh_syn\n"
                   f"w_inh = clip(w_inh + apre_inh*(wmax_inh_syn-w_inh), {W_INH_MIN}, wmax_inh_syn)")

    S_hh = Synapses(G_h, G_h, model=stdp_inh_model, on_pre=on_pre_inh, on_post=on_post_inh)
    S_hh.connect(i=_SRC_HH, j=_TGT_HH)
    src_hh = np.array(S_hh.i)
    tgt_hh = np.array(S_hh.j)
    S_hh.wmax_inh_syn  = _WMAX_INH_MATRIX[src_hh, tgt_hh]
    S_hh.Apre_inh_syn  = _APRE_INH_MATRIX[src_hh, tgt_hh]
    S_hh.Apost_inh_syn = _APOST_INH_MATRIX[src_hh, tgt_hh]
    S_hh.w_inh         = W_HH_INIT[src_hh, tgt_hh]

    # ── Count-only spike monitors (per-neuron firing over a run window) ─────────
    spike_in  = SpikeMonitor(G_in, record=False)
    spike_hid = SpikeMonitor(G_h,  record=False)

    # ── Vectorised L1 normalisation every 500 ms + snapshot collection ─────────
    wmax_syn_arr     = np.array(S_ih.wmax_syn)
    wmax_inh_syn_arr = np.array(S_hh.wmax_inh_syn)
    run_state = {"epoch": 0, "snapshots": []}

    @network_operation(dt=NORM_DT, when='end')
    def normalize_weights():
        # Excitatory: scale any column whose L1 exceeds NORM_LIMIT_EXC.
        w = np.array(S_ih.w)
        col_sum = np.bincount(tgt_ih, weights=w, minlength=N_H)
        scale = np.where(col_sum > NORM_LIMIT_EXC, NORM_LIMIT_EXC / col_sum, 1.0)
        S_ih.w[:] = np.clip(w * scale[tgt_ih], wmin, wmax_syn_arr)

        # Inhibitory: same, against NORM_LIMIT_INH.
        wi = np.array(S_hh.w_inh)
        col_sum_i = np.bincount(tgt_hh, weights=wi, minlength=N_H)
        scale_i = np.where(col_sum_i > NORM_LIMIT_INH, NORM_LIMIT_INH / col_sum_i, 1.0)
        S_hh.w_inh[:] = np.clip(wi * scale_i[tgt_hh], W_INH_MIN, wmax_inh_syn_arr)

        if run_state["epoch"] >= SNAPSHOT_FROM_EPOCH:
            w_snap = np.zeros((N_IN, N_H))
            w_snap[src_ih, tgt_ih] = np.array(S_ih.w)
            run_state["snapshots"].append(w_snap)

    net = Network(G_in, G_h, S_ih, S_hh, spike_in, spike_hid, normalize_weights)
    G_h.vth = vth_init
    net.store('init')   # clean snapshot: clock=0, v=0, a=0, vth=vth_init, monitors empty

    return SimpleNamespace(
        net=net, G_in=G_in, G_h=G_h, S_ih=S_ih, S_hh=S_hh,
        src_ih=src_ih, tgt_ih=tgt_ih, src_hh=src_hh, tgt_hh=tgt_hh,
        spike_in=spike_in, spike_hid=spike_hid, run_state=run_state,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Per-wav training
# ═══════════════════════════════════════════════════════════════════════════════

def train_fingerprint(h, wav_path):
    """Run NUM_EPOCHS over one wav; return (fp, in_counts, hid_counts) or None.

    fp         : (N_IN, N_H) float32 — mean of the final-epoch weight snapshots.
    in_counts  : (N_IN,) int — input spike counts over the final-epoch window.
    hid_counts : (N_H,)  int — hidden spike counts over the final-epoch window.

    net.restore('init') resets state AND the count-only monitors at the start of
    every epoch, so the counts read after the last run() reflect epoch 3 only —
    the same window the snapshots span.
    """
    try:
        I, T = compute_spike_input_current(wav_path, **ENCODER_KWARGS)
    except Exception as e:
        print(f"  [skip {wav_path}: {e}]")
        return None

    w_ih = W_IH_INIT.copy()
    w_hh = W_HH_INIT.copy()
    h.run_state["snapshots"] = []
    I_timed = TimedArray(I.T.astype(float), dt=DT_SIM)

    for epoch_idx in range(NUM_EPOCHS):
        h.run_state["epoch"] = epoch_idx
        h.net.restore('init')
        h.G_in.namespace["I_timed"] = I_timed

        h.S_ih.w     = w_ih[h.src_ih, h.tgt_ih]
        h.S_ih.apre  = 0
        h.S_ih.apost = 0
        h.S_hh.w_inh     = w_hh[h.src_hh, h.tgt_hh]
        h.S_hh.apre_inh  = 0
        h.S_hh.apost_inh = 0

        h.net.run(T * DT_SIM)

        w_ih = np.zeros((N_IN, N_H)); w_ih[h.src_ih, h.tgt_ih] = np.array(h.S_ih.w)
        w_hh = np.zeros((N_H, N_H));  w_hh[h.src_hh, h.tgt_hh] = np.array(h.S_hh.w_inh)

    snapshots = h.run_state["snapshots"]
    fp = (np.mean(np.stack(snapshots), axis=0).astype(np.float32)
          if snapshots else w_ih.astype(np.float32))
    in_counts  = np.array(h.spike_in.count,  dtype=np.int64)
    hid_counts = np.array(h.spike_hid.count, dtype=np.int64)
    return fp, in_counts, hid_counts


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor transform → ANN-ready per-sample output
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_activity(counts):
    """Per-sample [0,1] via ÷99th-percentile + clip. Preserves 0 (silent)."""
    denom = np.percentile(counts, 99)
    if denom <= 0:
        return np.zeros_like(counts, dtype=np.float16)
    return np.clip(counts / denom, 0.0, 1.0).astype(np.float16)


def fingerprint_to_sample(fp, in_counts, hid_counts):
    """(N_IN,N_H) weight matrix + spike counts → (weights, input_activity, hidden_activity).

    weights         (4,33,384) float16 — raw type-images, silent cells zeroed
    input_activity  (384,)     float16 — per-sample [0,1]
    hidden_activity (384,)     float16 — per-sample [0,1]
    """
    weights = fp[IN_IDX, _J_ROW].astype(np.float32)            # (4,33,N_H)

    # Zero any weight whose presynaptic input neuron OR hidden neuron never fired.
    in_silent  = (in_counts == 0)
    hid_silent = (hid_counts == 0)
    silent = in_silent[IN_IDX] | hid_silent[None, None, :]      # (4,33,N_H)
    weights[silent] = 0.0

    return (weights.astype(np.float16),
            _normalize_activity(in_counts),
            _normalize_activity(hid_counts))
