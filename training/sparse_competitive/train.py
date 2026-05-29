import numpy as np
np.random.seed(42)
from brian2 import *
import os
import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.recorder import Recorder

import time
start = time.time()

# ============================================================
# Dataset
# ============================================================
wav_files = [
    "datasets/vox1_single_person_nano_2/dev/00001.wav",
    "datasets/vox1_single_person_nano_2/dev/00002.wav"
]
print(f"Found {len(wav_files)} wav files")

EPOCHS   = 8
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Architecture constants
# ============================================================

N_CH            = 96            # frequency channels
NEURONS_PER_CH  = 7             # 4 sustained + 2 onset + 1 phase
N_IN            = 672           # N_CH * NEURONS_PER_CH
NEURONS_PER_GRP = 4             # hidden neurons per group

# -- CNN pathway (local spectral features) --
# Channels are bundled into non-overlapping units of 4; each CNN group reads 3
# consecutive units (stride 1, wraparound) → 12-channel / 84-input receptive field.
CNN_UNIT_SIZE  = 4
N_CNN_UNITS    = N_CH // CNN_UNIT_SIZE        # 24
CNN_RF_UNITS   = 3
N_CNN_GROUPS   = N_CNN_UNITS                   # 24 (one group per starting unit)
N_CNN          = N_CNN_GROUPS * NEURONS_PER_GRP  # 96
CNN_RF_INPUTS  = CNN_RF_UNITS * CNN_UNIT_SIZE * NEURONS_PER_CH  # 84

# -- Duo pathway (relational spectral features) --
# Channels are bundled into non-overlapping units of 8; the 12 units are split
# into even/odd partitions, and every unordered pair within a partition forms a
# Duo group → 16-channel / 112-input receptive field.
DUO_UNIT_SIZE  = 8
N_DUO_UNITS    = N_CH // DUO_UNIT_SIZE          # 12
N_DUO_GROUPS   = 30                             # 15 even + 15 odd
N_DUO          = N_DUO_GROUPS * NEURONS_PER_GRP  # 120
DUO_RF_INPUTS  = 2 * DUO_UNIT_SIZE * NEURONS_PER_CH  # 112

DT_SIM = 1 * ms

# ============================================================
# Hyperparameters
# ============================================================

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 100 * ms
tau_current = 1 * ms
beta        = 1.0
v_th_in     = 1.0

# -- Hidden layer (adaptive-threshold LIF, shared CNN + Duo) --
tau_h       = 50 * ms
tau_vth     = 100 * ms
vth_rest    = 0.8
vth_init    = 0.8
vth_jump    = 0.3
tau_r       = 10 * ms
sigma_noise = 0.03 * second**(-0.5)

# -- STDP trace decay (shared) --
taupre  = 20 * ms
taupost = 20 * ms

# -- Excitatory STDP (input → hidden) --
APRE_EXC  =  0.002
APOST_EXC = -0.0024
WMAX_EXC  =  1.0
WMIN_EXC  =  0.0
W_INIT_SUM = 2.0

# -- Inhibitory lateral STDP (multiplicative, slower than excitatory) --
# inhibitory_lr = 0.25 * excitatory_lr
APRE_INH  =  0.0005
APOST_INH = -0.0006
WMIN_INH  =  0.0
WMAX_INH  =  1.0    # global ceiling; effective per-synapse ceiling = WMAX_INH * J

# -- Jaccard similarity-based lateral inhibition --
# Trace increments and weight ceiling both scale with J:
#   apre_inh  += APRE_INH  * J  (per pre-spike)
#   apost_inh += APOST_INH * J  (per post-spike)
#   ceiling    = WMAX_INH  * J  (per synapse)
# LTP at w=0 scales as WMAX_INH * J^2; LTD scales as J * w.
SIMILARITY_THRESHOLD = 0.0

# -- Homeostatic normalisation (excitatory only) --
NORM_LIMIT_EXC = 2.0

# ============================================================
# Connectivity precomputation
# ============================================================

print("Precomputing connectivity...")
t_conn = time.time()

# ── Spectral units → channel sets ─────────────────────────────────────────────
# CNN unit u  = channels [4u, 4u+4);  Duo unit u = channels [8u, 8u+8)
def _cnn_unit_channels(u):
    return range(u * CNN_UNIT_SIZE, (u + 1) * CNN_UNIT_SIZE)

def _duo_unit_channels(u):
    return range(u * DUO_UNIT_SIZE, (u + 1) * DUO_UNIT_SIZE)

# CNN group k reads CNN_RF_UNITS consecutive units {k, k+1, k+2} (mod N_CNN_UNITS)
cnn_group_ch = []
for k in range(N_CNN_GROUPS):
    chans = []
    for d in range(CNN_RF_UNITS):
        chans.extend(_cnn_unit_channels((k + d) % N_CNN_UNITS))
    cnn_group_ch.append(frozenset(chans))

# Duo groups: split units into even/odd partitions, take every unordered pair
even_units = list(range(0, N_DUO_UNITS, 2))   # [0, 2, 4, 6, 8, 10]
odd_units  = list(range(1, N_DUO_UNITS, 2))    # [1, 3, 5, 7, 9, 11]
duo_unit_pairs = (list(itertools.combinations(even_units, 2)) +
                  list(itertools.combinations(odd_units, 2)))   # 15 + 15 = 30
duo_group_ch = []
for (ua, ub) in duo_unit_pairs:
    duo_group_ch.append(frozenset(list(_duo_unit_channels(ua)) +
                                  list(_duo_unit_channels(ub))))

# ── Input → CNN ─────────────────────────────────────────────────────────────
# Each CNN group: 12 channels × 7 inputs × 4 outputs = 336 synapses per group.
# Loop order group→sorted channels→inputs→outputs keeps the regular layout that
# the vectorised excitatory normalisation index trick relies on.
_cnn_src, _cnn_tgt = [], []
for k in range(N_CNN_GROUPS):
    for ch in sorted(cnn_group_ch[k]):
        for n_in in range(ch * NEURONS_PER_CH, (ch + 1) * NEURONS_PER_CH):
            for n_out in range(k * NEURONS_PER_GRP, (k + 1) * NEURONS_PER_GRP):
                _cnn_src.append(n_in)
                _cnn_tgt.append(n_out)
cnn_src_ih = np.array(_cnn_src, dtype=np.int32)
cnn_tgt_ih = np.array(_cnn_tgt, dtype=np.int32)

# ── Input → Duo ─────────────────────────────────────────────────────────────
# Each Duo group: 16 channels × 7 inputs × 4 outputs = 448 synapses per group.
_duo_src, _duo_tgt = [], []
for g in range(N_DUO_GROUPS):
    for ch in sorted(duo_group_ch[g]):
        for n_in in range(ch * NEURONS_PER_CH, (ch + 1) * NEURONS_PER_CH):
            for n_out in range(g * NEURONS_PER_GRP, (g + 1) * NEURONS_PER_GRP):
                _duo_src.append(n_in)
                _duo_tgt.append(n_out)
duo_src_ih = np.array(_duo_src, dtype=np.int32)
duo_tgt_ih = np.array(_duo_tgt, dtype=np.int32)

# ── Lateral inhibition: Jaccard similarity (pathway-agnostic) ─────────────────
# Helper: add all 4×4 directed connections for a group pair
def _add_pairs(a_grp, b_grp, wmax_val, src_list, tgt_list, wmax_list, n_per_grp=NEURONS_PER_GRP):
    a_ns = np.arange(a_grp * n_per_grp, (a_grp + 1) * n_per_grp)
    b_ns = np.arange(b_grp * n_per_grp, (b_grp + 1) * n_per_grp)
    srcs = np.repeat(a_ns, n_per_grp)
    tgts = np.tile(b_ns, n_per_grp)
    src_list.append(srcs);  tgt_list.append(tgts)
    wmax_list.append(np.full(n_per_grp * n_per_grp, wmax_val))

def _jaccard(a, b):
    inter = len(a & b)
    return inter / len(a | b) if inter > 0 else 0.0

lat_cc_s, lat_cc_t, lat_cc_w = [], [], []
lat_dd_s, lat_dd_t, lat_dd_w = [], [], []
lat_cd_s, lat_cd_t, lat_cd_w = [], [], []
lat_dc_s, lat_dc_t, lat_dc_w = [], [], []

# Within-group: identical receptive field ⇒ J = 1 = J_max ⇒ wmax = 1.0 (i≠j).
# Use ni/nj to avoid leaking loop vars into Brian2's run namespace (conflicts with
# Brian2's internal 'i' = pre-synaptic index, causing a resolution warning).
for k in range(N_CNN_GROUPS):
    ns = np.arange(k * NEURONS_PER_GRP, (k + 1) * NEURONS_PER_GRP)
    for ni in ns:
        for nj in ns:
            if ni != nj:
                lat_cc_s.append(np.array([ni])); lat_cc_t.append(np.array([nj]))
                lat_cc_w.append(np.array([1.0]))

for g in range(N_DUO_GROUPS):
    ns = np.arange(g * NEURONS_PER_GRP, (g + 1) * NEURONS_PER_GRP)
    for ni in ns:
        for nj in ns:
            if ni != nj:
                lat_dd_s.append(np.array([ni])); lat_dd_t.append(np.array([nj]))
                lat_dd_w.append(np.array([1.0]))

# CNN–CNN cross pairs
for k in range(N_CNN_GROUPS):
    for m in range(k + 1, N_CNN_GROUPS):
        J = _jaccard(cnn_group_ch[k], cnn_group_ch[m])
        if J > SIMILARITY_THRESHOLD:
            _add_pairs(k, m, J, lat_cc_s, lat_cc_t, lat_cc_w)
            _add_pairs(m, k, J, lat_cc_s, lat_cc_t, lat_cc_w)

# Duo–Duo cross pairs
for g in range(N_DUO_GROUPS):
    for h in range(g + 1, N_DUO_GROUPS):
        J = _jaccard(duo_group_ch[g], duo_group_ch[h])
        if J > SIMILARITY_THRESHOLD:
            _add_pairs(g, h, J, lat_dd_s, lat_dd_t, lat_dd_w)
            _add_pairs(h, g, J, lat_dd_s, lat_dd_t, lat_dd_w)

# CNN–Duo cross pairs (CNN→Duo in cd, Duo→CNN in dc — bidirectional)
for k in range(N_CNN_GROUPS):
    for g in range(N_DUO_GROUPS):
        J = _jaccard(cnn_group_ch[k], duo_group_ch[g])
        if J > SIMILARITY_THRESHOLD:
            _add_pairs(k, g, J, lat_cd_s, lat_cd_t, lat_cd_w)
            _add_pairs(g, k, J, lat_dc_s, lat_dc_t, lat_dc_w)

def _cat_idx(lst):
    return np.concatenate(lst).astype(np.int32) if lst else np.array([], dtype=np.int32)

def _cat_w(lst):
    return np.concatenate(lst).astype(float) if lst else np.array([], dtype=float)

lat_cc_src, lat_cc_tgt, lat_cc_wmax = _cat_idx(lat_cc_s), _cat_idx(lat_cc_t), _cat_w(lat_cc_w)
lat_dd_src, lat_dd_tgt, lat_dd_wmax = _cat_idx(lat_dd_s), _cat_idx(lat_dd_t), _cat_w(lat_dd_w)
lat_cd_src, lat_cd_tgt, lat_cd_wmax = _cat_idx(lat_cd_s), _cat_idx(lat_cd_t), _cat_w(lat_cd_w)
lat_dc_src, lat_dc_tgt, lat_dc_wmax = _cat_idx(lat_dc_s), _cat_idx(lat_dc_t), _cat_w(lat_dc_w)

print(f"  S_in_cnn : {len(cnn_src_ih):>10,} synapses")
print(f"  S_in_duo : {len(duo_src_ih):>10,} synapses")
print(f"  S_lat_cc : {len(lat_cc_src):>10,} synapses")
print(f"  S_lat_dd : {len(lat_dd_src):>10,} synapses")
print(f"  S_lat_cd : {len(lat_cd_src):>10,} synapses")
print(f"  S_lat_dc : {len(lat_dc_src):>10,} synapses")
print(f"  Connectivity precomputed in {time.time()-t_conn:.1f}s")

# ============================================================
# Weight initialisation
# ============================================================

def _init_exc_flat(tgt_arr, n_post, n_syn, w_sum, wmax, wmin, rng):
    w = rng.uniform(wmin + 1e-4, wmax, size=n_syn)
    for j in range(n_post):
        idx = np.where(tgt_arr == j)[0]
        if len(idx) == 0:
            continue
        s = w[idx].sum()
        if s > 0:
            w[idx] = np.clip(w[idx] / s * w_sum, wmin, wmax)
    return w

_rng = np.random.default_rng(42)
w_in_cnn = _init_exc_flat(cnn_tgt_ih, N_CNN, len(cnn_src_ih), W_INIT_SUM, WMAX_EXC, WMIN_EXC, _rng)
w_in_duo = _init_exc_flat(duo_tgt_ih, N_DUO, len(duo_src_ih), W_INIT_SUM, WMAX_EXC, WMIN_EXC, _rng)
w_lat_cc = np.zeros(len(lat_cc_src))
w_lat_dd = np.zeros(len(lat_dd_src))
w_lat_cd = np.zeros(len(lat_cd_src))
w_lat_dc = np.zeros(len(lat_dc_src))

# Snapshot the initial flat excitatory weights (used for epoch-0 init weight matrices)
w_init_cnn, w_init_duo = w_in_cnn.copy(), w_in_duo.copy()
# w_init_lat captured below, after inhibitory weights are warm-started

# ============================================================
# Build Brian2 network
# ============================================================

defaultclock.dt = DT_SIM

# ── Input neurons ─────────────────────────────────────────────────────────────
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

# ── Hidden neurons (CNN and Duo share equations) ──────────────────────────────
eqs_h = f"""
dv/dt       = -v / tau_h + sigma_noise * xi   : 1
dvth/dt     = -(vth - {vth_rest}) / tau_vth   : 1
dtrace_r/dt = -trace_r / tau_r                : 1
"""
_hid_kw = dict(
    threshold="v > vth",
    reset=f"v=0; vth=vth+{vth_jump}; trace_r=1;",
    method="euler"
)
G_cnn = NeuronGroup(N_CNN, eqs_h, **_hid_kw)
G_duo = NeuronGroup(N_DUO, eqs_h, **_hid_kw)

# ── Excitatory STDP model ─────────────────────────────────────────────────────
_exc_model = """
w         : 1
dapre/dt  = -apre  / taupre  : 1 (event-driven)
dapost/dt = -apost / taupost : 1 (event-driven)
"""
_on_pre_exc  = (f"v_post += w * (1 - trace_r_post)\n"
                f"apre  += {APRE_EXC}\n"
                f"w = clip(w + apost*(w - {WMIN_EXC}), {WMIN_EXC}, {WMAX_EXC})")
_on_post_exc = (f"apost += {APOST_EXC}\n"
                f"w = clip(w + apre*({WMAX_EXC} - w), {WMIN_EXC}, {WMAX_EXC})")

S_in_cnn = Synapses(G_in, G_cnn, model=_exc_model, on_pre=_on_pre_exc, on_post=_on_post_exc)
S_in_cnn.connect(i=cnn_src_ih, j=cnn_tgt_ih)

S_in_duo = Synapses(G_in, G_duo, model=_exc_model, on_pre=_on_pre_exc, on_post=_on_post_exc)
S_in_duo.connect(i=duo_src_ih, j=duo_tgt_ih)

# ── Inhibitory lateral STDP model ─────────────────────────────────────────────
_inh_model = """
w_inh          : 1
dapre_inh/dt   = -apre_inh  / taupre  : 1 (event-driven)
dapost_inh/dt  = -apost_inh / taupost : 1 (event-driven)
jaccard_syn    : 1
"""
_on_pre_inh  = (f"v_post -= w_inh\n"
                f"apre_inh += {APRE_INH} * jaccard_syn\n"
                f"w_inh = clip(w_inh + apost_inh*(w_inh - {WMIN_INH}), {WMIN_INH}, {WMAX_INH}*jaccard_syn)")
_on_post_inh = (f"apost_inh += {APOST_INH} * jaccard_syn\n"
                f"w_inh = clip(w_inh + apre_inh*({WMAX_INH}*jaccard_syn - w_inh), {WMIN_INH}, {WMAX_INH}*jaccard_syn)")

def _make_inh_syn(pre, post, src_arr, tgt_arr, jaccard_arr):
    S = Synapses(pre, post, model=_inh_model, on_pre=_on_pre_inh, on_post=_on_post_inh)
    S.connect(i=src_arr, j=tgt_arr)
    S.jaccard_syn = jaccard_arr
    S.w_inh = 0.0
    return S

S_lat_cc = _make_inh_syn(G_cnn, G_cnn, lat_cc_src, lat_cc_tgt, lat_cc_wmax)
S_lat_dd = _make_inh_syn(G_duo, G_duo, lat_dd_src, lat_dd_tgt, lat_dd_wmax)
S_lat_cd = _make_inh_syn(G_cnn, G_duo, lat_cd_src, lat_cd_tgt, lat_cd_wmax)
S_lat_dc = _make_inh_syn(G_duo, G_cnn, lat_dc_src, lat_dc_tgt, lat_dc_wmax)

net = Network(G_in, G_cnn, G_duo,
              S_in_cnn, S_in_duo,
              S_lat_cc, S_lat_dd, S_lat_cd, S_lat_dc)

# ============================================================
# Recorder setup
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",      G_in)
recorder.track_group("cnn_hidden", G_cnn)
recorder.track_group("duo_hidden", G_duo)

recorder.track_synapses("in->cnn",  S_in_cnn,  cnn_src_ih,  cnn_tgt_ih)
recorder.track_synapses("in->duo",  S_in_duo,  duo_src_ih,  duo_tgt_ih)
recorder.track_synapses("lat_cc",   S_lat_cc,  lat_cc_src,  lat_cc_tgt,  weight_var="w_inh")
recorder.track_synapses("lat_dd",   S_lat_dd,  lat_dd_src,  lat_dd_tgt,  weight_var="w_inh")
recorder.track_synapses("lat_cd",   S_lat_cd,  lat_cd_src,  lat_cd_tgt,  weight_var="w_inh")
recorder.track_synapses("lat_dc",   S_lat_dc,  lat_dc_src,  lat_dc_tgt,  weight_var="w_inh")

recorder.build()

# ============================================================
# Normalisation index precomputation
# ============================================================

_CNN_FANIN = CNN_RF_UNITS * CNN_UNIT_SIZE * NEURONS_PER_CH   # 84 incoming / CNN neuron
_DUO_FANIN = 2 * DUO_UNIT_SIZE * NEURONS_PER_CH              # 112 incoming / Duo neuron

# Excitatory CNN: synapse order group→channel→input→output means neuron k*4+off
# owns indices k*(_CNN_FANIN*4) + off + 4*m, m = 0.._CNN_FANIN-1.
_cnn_base = (np.arange(N_CNN) // NEURONS_PER_GRP) * (_CNN_FANIN * NEURONS_PER_GRP)
_cnn_off  = np.arange(N_CNN) % NEURONS_PER_GRP
cnn_exc_idx = _cnn_base[:, None] + _cnn_off[:, None] + NEURONS_PER_GRP * np.arange(_CNN_FANIN)[None, :]

# Excitatory Duo: same regular layout with _DUO_FANIN incoming per neuron.
_duo_base = (np.arange(N_DUO) // NEURONS_PER_GRP) * (_DUO_FANIN * NEURONS_PER_GRP)
_duo_off  = np.arange(N_DUO) % NEURONS_PER_GRP
duo_exc_idx = _duo_base[:, None] + _duo_off[:, None] + NEURONS_PER_GRP * np.arange(_DUO_FANIN)[None, :]

# Capture zero-init inhibitory weights as the epoch-0 init snapshot
w_init_lat = (w_lat_cc.copy(), w_lat_dd.copy(), w_lat_cd.copy(), w_lat_dc.copy())


@network_operation(dt=500 * ms, when='end')
def normalize_weights():
    # ── Excitatory CNN (vectorised) ───────────────────────────────────────────
    w_ec = np.array(S_in_cnn.w)
    w_mat = w_ec[cnn_exc_idx]         # (384, 21)
    wsums = w_mat.sum(axis=1)          # (384,)
    mask  = wsums > NORM_LIMIT_EXC
    if mask.any():
        scales = np.where(mask, NORM_LIMIT_EXC / np.where(wsums > 0, wsums, 1.0), 1.0)
        w_mat_new = np.clip(w_mat * scales[:, None], WMIN_EXC, WMAX_EXC)
        w_ec[cnn_exc_idx[mask].ravel()] = w_mat_new[mask].ravel()
        S_in_cnn.w = w_ec

    # ── Excitatory Duo (vectorised) ───────────────────────────────────────────
    w_ed = np.array(S_in_duo.w)
    w_mat = w_ed[duo_exc_idx]          # (18240, 14)
    wsums = w_mat.sum(axis=1)
    mask  = wsums > NORM_LIMIT_EXC
    if mask.any():
        scales = np.where(mask, NORM_LIMIT_EXC / np.where(wsums > 0, wsums, 1.0), 1.0)
        w_mat_new = np.clip(w_mat * scales[:, None], WMIN_EXC, WMAX_EXC)
        w_ed[duo_exc_idx[mask].ravel()] = w_mat_new[mask].ravel()
        S_in_duo.w = w_ed



net.add(normalize_weights)

G_cnn.vth = vth_init
G_duo.vth = vth_init
net.store('init')

# ============================================================
# Helper: build sparse weight matrix (used in recorder calls)
# ============================================================

def _to_matrix(src, tgt, w_flat, n_pre, n_post):
    m = np.zeros((n_pre, n_post), dtype=np.float32)
    m[src, tgt] = w_flat
    return m


# ============================================================
# Training loop
# ============================================================

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_idx}/{EPOCHS - 1}")
    print(f"{'='*60}")

    recorder.reset_epoch()

    for sample_idx, audio_path in enumerate(wav_files):
        print(f"  [epoch {epoch_idx}, sample {sample_idx}/{len(wav_files)-1}] "
              f"{os.path.relpath(audio_path)}")

        try:
            I, T = compute_spike_input_current(
                audio_path,
                scale=0.8,
                sustained_per_band=4,
                onset_per_band=2,
                phase_per_band=1,
                sust_spread_min=0.7,
                sust_spread_max=1.3,
                num_filters=96,
            )
        except Exception as e:
            print(f"    Error encoding audio: {e}")
            continue

        duration_s = float(T) * float(DT_SIM)

        # ── Reset and inject current ───────────────────────────────────────────
        net.restore('init')
        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Restore learned weights (override the stored initial values)
        S_in_cnn.w    = w_in_cnn;  S_in_cnn.apre  = 0;  S_in_cnn.apost = 0
        S_in_duo.w    = w_in_duo;  S_in_duo.apre  = 0;  S_in_duo.apost = 0
        S_lat_cc.w_inh = w_lat_cc; S_lat_cc.apre_inh = 0; S_lat_cc.apost_inh = 0
        S_lat_dd.w_inh = w_lat_dd; S_lat_dd.apre_inh = 0; S_lat_dd.apost_inh = 0
        S_lat_cd.w_inh = w_lat_cd; S_lat_cd.apre_inh = 0; S_lat_cd.apost_inh = 0
        S_lat_dc.w_inh = w_lat_dc; S_lat_dc.apre_inh = 0; S_lat_dc.apost_inh = 0

        recorder._elapsed_ms = 0.0
        recorder.before_sample(sample_idx)

        # ── Simulate ───────────────────────────────────────────────────────────
        net.run(T * DT_SIM)

        # ── Extract updated weights ────────────────────────────────────────────
        w_in_cnn = np.array(S_in_cnn.w)
        w_in_duo = np.array(S_in_duo.w)
        w_lat_cc = np.array(S_lat_cc.w_inh)
        w_lat_dd = np.array(S_lat_dd.w_inh)
        w_lat_cd = np.array(S_lat_cd.w_inh)
        w_lat_dc = np.array(S_lat_dc.w_inh)

        # ── Record: after ──────────────────────────────────────────────────────
        # All matrices are small now (max 672×120) — record every synapse group.
        all_w_matrices = {
            "in->cnn": _to_matrix(cnn_src_ih, cnn_tgt_ih, w_in_cnn, N_IN,  N_CNN),
            "in->duo": _to_matrix(duo_src_ih, duo_tgt_ih, w_in_duo, N_IN,  N_DUO),
            "lat_cc":  _to_matrix(lat_cc_src, lat_cc_tgt, w_lat_cc, N_CNN, N_CNN),
            "lat_dd":  _to_matrix(lat_dd_src, lat_dd_tgt, w_lat_dd, N_DUO, N_DUO),
            "lat_cd":  _to_matrix(lat_cd_src, lat_cd_tgt, w_lat_cd, N_CNN, N_DUO),
            "lat_dc":  _to_matrix(lat_dc_src, lat_dc_tgt, w_lat_dc, N_DUO, N_CNN),
        }
        recorder.after_sample(sample_idx, duration_s, w_matrices=all_w_matrices)

    # ── End of epoch ──────────────────────────────────────────────────────────
    init_weights = None
    if epoch_idx == 0:
        _ic, _id, _il = w_init_cnn, w_init_duo, w_init_lat
        init_weights = {
            "in->cnn": _to_matrix(cnn_src_ih, cnn_tgt_ih, _ic,    N_IN,  N_CNN),
            "in->duo": _to_matrix(duo_src_ih, duo_tgt_ih, _id,    N_IN,  N_DUO),
            "lat_cc":  _to_matrix(lat_cc_src, lat_cc_tgt, _il[0], N_CNN, N_CNN),
            "lat_dd":  _to_matrix(lat_dd_src, lat_dd_tgt, _il[1], N_DUO, N_DUO),
            "lat_cd":  _to_matrix(lat_cd_src, lat_cd_tgt, _il[2], N_CNN, N_DUO),
            "lat_dc":  _to_matrix(lat_dc_src, lat_dc_tgt, _il[3], N_DUO, N_CNN),
        }
    recorder.save_epoch(
        epoch_idx,
        save_dir=SAVE_DIR,
        final_weights=all_w_matrices,
        init_weights=init_weights,
    )


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
