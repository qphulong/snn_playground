import numpy as np
np.random.seed(42)
from brian2 import *
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.spike_encoding import compute_spike_input_current
from src.recorder import Recorder

import time
start = time.time()

# ============================================================
# Dataset
# ============================================================
wav_files = [
    "datasets/vox1_single_person_nano_2/dev/00008.wav",
    "datasets/vox1_10person_fingerprint/wav_dev/id10397/id10397_00002/00003.wav",
    "datasets/vox1_cleaned/wav_dev/id10008/id10008_00009/00002.wav",
    "datasets/vox1_cleaned/wav_dev/id10019/id10019_00005/00003.wav",
    "datasets/vox1_cleaned/wav_dev/id11251/id11251_00005/00001.wav"
]
print(f"Found {len(wav_files)} wav files")

EPOCHS   = 1
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Architecture constants
# ============================================================

N_CH            = 96            # frequency channels
NEURONS_PER_CH  = 7             # 4 sustained + 2 onset + 1 phase
N_IN            = 672           # N_CH * NEURONS_PER_CH
NEURONS_PER_GRP = 4             # hidden neurons per group

N_DUO_GROUPS = N_CH * (N_CH - 1) // 2   # 4560
N_CNN        = N_CH * NEURONS_PER_GRP    # 384
N_DUO        = N_DUO_GROUPS * NEURONS_PER_GRP  # 18240

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

# -- Inhibitory lateral STDP --
APRE_INH  =  0.002
APOST_INH = -0.0024
WMIN_INH  =  0.0
WMAX_L3   =  1.0   # within-group
WMAX_L2   =  0.6   # 2 common channels
WMAX_L1   =  0.3   # 1 common channel

# -- Homeostatic normalisation (separate limits) --
NORM_LIMIT_EXC = 2.0
NORM_LIMIT_INH = 2.0

# ============================================================
# Connectivity precomputation
# ============================================================

print("Precomputing connectivity...")
t_conn = time.time()

duo_pairs        = [(x, y) for x in range(N_CH) for y in range(x + 1, N_CH)]
cnn_channels     = [frozenset([(k - 1) % N_CH, k, (k + 1) % N_CH]) for k in range(N_CH)]
duo_channels     = [frozenset(p) for p in duo_pairs]

# ── Input → CNN ─────────────────────────────────────────────────────────────
# Each CNN group k: 3 channels × 7 inputs × 4 outputs = 84 synapses per group
_cnn_src, _cnn_tgt = [], []
for k in range(N_CH):
    for ch in sorted(cnn_channels[k]):
        for n_in in range(ch * NEURONS_PER_CH, (ch + 1) * NEURONS_PER_CH):
            for n_out in range(k * NEURONS_PER_GRP, (k + 1) * NEURONS_PER_GRP):
                _cnn_src.append(n_in)
                _cnn_tgt.append(n_out)
cnn_src_ih = np.array(_cnn_src, dtype=np.int32)
cnn_tgt_ih = np.array(_cnn_tgt, dtype=np.int32)

# ── Input → Duo ─────────────────────────────────────────────────────────────
# Each Duo group: 2 channels × 7 inputs × 4 outputs = 56 synapses per group
_duo_src, _duo_tgt = [], []
for idx, (x, y) in enumerate(duo_pairs):
    for ch in [x, y]:
        for n_in in range(ch * NEURONS_PER_CH, (ch + 1) * NEURONS_PER_CH):
            for n_out in range(idx * NEURONS_PER_GRP, (idx + 1) * NEURONS_PER_GRP):
                _duo_src.append(n_in)
                _duo_tgt.append(n_out)
duo_src_ih = np.array(_duo_src, dtype=np.int32)
duo_tgt_ih = np.array(_duo_tgt, dtype=np.int32)

# ── Lateral inhibition ───────────────────────────────────────────────────────
# Helper: add all 4×4 directed connections for a group pair
def _add_pairs(a_grp, b_grp, wmax_val, src_list, tgt_list, wmax_list, n_per_grp=NEURONS_PER_GRP):
    a_ns = np.arange(a_grp * n_per_grp, (a_grp + 1) * n_per_grp)
    b_ns = np.arange(b_grp * n_per_grp, (b_grp + 1) * n_per_grp)
    srcs = np.repeat(a_ns, n_per_grp)
    tgts = np.tile(b_ns, n_per_grp)
    src_list.append(srcs);  tgt_list.append(tgts)
    wmax_list.append(np.full(n_per_grp * n_per_grp, wmax_val))

lat_cc_s, lat_cc_t, lat_cc_w = [], [], []
lat_dd_s, lat_dd_t, lat_dd_w = [], [], []
lat_cd_s, lat_cd_t, lat_cd_w = [], [], []
lat_dc_s, lat_dc_t, lat_dc_w = [], [], []

# Level 3: within-group (i≠j)
# Use ni/nj to avoid leaking loop vars into Brian2's run namespace (conflicts with
# Brian2's internal 'i' = pre-synaptic index, causing a resolution warning).
for k in range(N_CH):
    ns = np.arange(k * NEURONS_PER_GRP, (k + 1) * NEURONS_PER_GRP)
    for ni in ns:
        for nj in ns:
            if ni != nj:
                lat_cc_s.append(np.array([ni])); lat_cc_t.append(np.array([nj]))
                lat_cc_w.append(np.array([WMAX_L3]))

for idx in range(N_DUO_GROUPS):
    ns = np.arange(idx * NEURONS_PER_GRP, (idx + 1) * NEURONS_PER_GRP)
    for ni in ns:
        for nj in ns:
            if ni != nj:
                lat_dd_s.append(np.array([ni])); lat_dd_t.append(np.array([nj]))
                lat_dd_w.append(np.array([WMAX_L3]))

# CNN–CNN: all unique pairs (k < m), connection level by channel-set intersection.
# Using absolute pair indices avoids the wraparound dedup bug that k<(k+1)%N_CH has.
for k in range(N_CH):
    for m in range(k + 1, N_CH):
        n_common = len(cnn_channels[k] & cnn_channels[m])
        if n_common == 2:
            _add_pairs(k, m, WMAX_L2, lat_cc_s, lat_cc_t, lat_cc_w)
            _add_pairs(m, k, WMAX_L2, lat_cc_s, lat_cc_t, lat_cc_w)
        elif n_common == 1:
            _add_pairs(k, m, WMAX_L1, lat_cc_s, lat_cc_t, lat_cc_w)
            _add_pairs(m, k, WMAX_L1, lat_cc_s, lat_cc_t, lat_cc_w)
        # n_common == 0: no connection

# CNN–Duo level 2 and 1
for k in range(N_CH):
    cnn_set = cnn_channels[k]
    for idx, duo_set in enumerate(duo_channels):
        n_common = len(cnn_set & duo_set)
        if n_common == 2:
            _add_pairs(k, idx, WMAX_L2, lat_cd_s, lat_cd_t, lat_cd_w)
            _add_pairs(idx, k, WMAX_L2, lat_dc_s, lat_dc_t, lat_dc_w)
        elif n_common == 1:
            _add_pairs(k, idx, WMAX_L1, lat_cd_s, lat_cd_t, lat_cd_w)
            _add_pairs(idx, k, WMAX_L1, lat_dc_s, lat_dc_t, lat_dc_w)

# Duo–Duo level 1: enumerate pairs sharing exactly 1 channel via channel index
channel_to_duo = [[] for _ in range(N_CH)]
for idx, (x, y) in enumerate(duo_pairs):
    channel_to_duo[x].append(idx)
    channel_to_duo[y].append(idx)

for ch in range(N_CH):
    grps = np.array(channel_to_duo[ch], dtype=np.int32)
    if len(grps) < 2:
        continue
    ia, ib = np.triu_indices(len(grps), k=1)
    a_grps = grps[ia];  b_grps = grps[ib]
    a_ns = a_grps[:, None] * NEURONS_PER_GRP + np.arange(NEURONS_PER_GRP)[None, :]  # (n,4)
    b_ns = b_grps[:, None] * NEURONS_PER_GRP + np.arange(NEURONS_PER_GRP)[None, :]  # (n,4)
    # A→B
    src_ab = np.repeat(a_ns, NEURONS_PER_GRP, axis=1).reshape(-1)
    tgt_ab = np.tile(b_ns, (1, NEURONS_PER_GRP)).reshape(-1)
    # B→A
    src_ba = np.repeat(b_ns, NEURONS_PER_GRP, axis=1).reshape(-1)
    tgt_ba = np.tile(a_ns, (1, NEURONS_PER_GRP)).reshape(-1)
    lat_dd_s.append(np.concatenate([src_ab, src_ba]))
    lat_dd_t.append(np.concatenate([tgt_ab, tgt_ba]))
    lat_dd_w.append(np.full(len(src_ab) + len(src_ba), WMAX_L1))

lat_cc_src = np.concatenate(lat_cc_s).astype(np.int32)
lat_cc_tgt = np.concatenate(lat_cc_t).astype(np.int32)
lat_cc_wmax = np.concatenate(lat_cc_w)
lat_dd_src = np.concatenate(lat_dd_s).astype(np.int32)
lat_dd_tgt = np.concatenate(lat_dd_t).astype(np.int32)
lat_dd_wmax = np.concatenate(lat_dd_w)
lat_cd_src = np.concatenate(lat_cd_s).astype(np.int32)
lat_cd_tgt = np.concatenate(lat_cd_t).astype(np.int32)
lat_cd_wmax = np.concatenate(lat_cd_w)
lat_dc_src = np.concatenate(lat_dc_s).astype(np.int32)
lat_dc_tgt = np.concatenate(lat_dc_t).astype(np.int32)
lat_dc_wmax = np.concatenate(lat_dc_w)

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
wmax_inh_syn   : 1
"""
_on_pre_inh  = (f"v_post -= w_inh\n"
                f"apre_inh += {APRE_INH}\n"
                f"w_inh = clip(w_inh + apost_inh*(w_inh - {WMIN_INH}), {WMIN_INH}, wmax_inh_syn)")
_on_post_inh = (f"apost_inh += {APOST_INH}\n"
                f"w_inh = clip(w_inh + apre_inh*(wmax_inh_syn - w_inh), {WMIN_INH}, wmax_inh_syn)")

def _make_inh_syn(pre, post, src_arr, tgt_arr, wmax_arr):
    S = Synapses(pre, post, model=_inh_model, on_pre=_on_pre_inh, on_post=_on_post_inh)
    S.connect(i=src_arr, j=tgt_arr)
    S.wmax_inh_syn = wmax_arr
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

print("Building normalisation index caches...")

# Excitatory CNN: each CNN neuron has exactly 21 incoming (3 ch × 7 neurons)
# Synapse order: for group k, block starts at k*84; neuron k*4+off at k*84+off+4*m, m=0..20
_cnn_base = (np.arange(N_CNN) // NEURONS_PER_GRP) * (3 * NEURONS_PER_CH * NEURONS_PER_GRP)
_cnn_off  = np.arange(N_CNN) % NEURONS_PER_GRP
cnn_exc_idx = _cnn_base[:, None] + _cnn_off[:, None] + NEURONS_PER_GRP * np.arange(3 * NEURONS_PER_CH)[None, :]

# Excitatory Duo: each Duo neuron has exactly 14 incoming (2 ch × 7 neurons)
# Synapse order: for group idx, block starts at idx*56; neuron idx*4+off at idx*56+off+4*m, m=0..13
_duo_base = (np.arange(N_DUO) // NEURONS_PER_GRP) * (2 * NEURONS_PER_CH * NEURONS_PER_GRP)
_duo_off  = np.arange(N_DUO) % NEURONS_PER_GRP
duo_exc_idx = _duo_base[:, None] + _duo_off[:, None] + NEURONS_PER_GRP * np.arange(2 * NEURONS_PER_CH)[None, :]

# Inhibitory: sorted-target approach for fast per-neuron index lookup
def _build_bounds(tgt_arr, n_post):
    order  = np.argsort(tgt_arr, kind='stable')
    bounds = np.searchsorted(tgt_arr[order], np.arange(n_post + 1))
    return order, bounds

_ord_cc, _bnd_cc = _build_bounds(lat_cc_tgt, N_CNN)
_ord_dc, _bnd_dc = _build_bounds(lat_dc_tgt, N_CNN)
_ord_dd, _bnd_dd = _build_bounds(lat_dd_tgt, N_DUO)
_ord_cd, _bnd_cd = _build_bounds(lat_cd_tgt, N_DUO)


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

    # ── Inhibitory CNN ────────────────────────────────────────────────────────
    w_cc = np.array(S_lat_cc.w_inh)
    w_dc = np.array(S_lat_dc.w_inh)
    changed_cc = np.zeros(len(w_cc), dtype=bool)
    changed_dc = np.zeros(len(w_dc), dtype=bool)
    for j in range(N_CNN):
        idx_cc = _ord_cc[_bnd_cc[j]:_bnd_cc[j + 1]]
        idx_dc = _ord_dc[_bnd_dc[j]:_bnd_dc[j + 1]]
        wsum = (w_cc[idx_cc].sum() if len(idx_cc) > 0 else 0.0) + \
               (w_dc[idx_dc].sum() if len(idx_dc) > 0 else 0.0)
        if wsum > NORM_LIMIT_INH:
            scale = NORM_LIMIT_INH / wsum
            if len(idx_cc) > 0:
                w_cc[idx_cc] = np.clip(w_cc[idx_cc] * scale, WMIN_INH, lat_cc_wmax[idx_cc])
                changed_cc[idx_cc] = True
            if len(idx_dc) > 0:
                w_dc[idx_dc] = np.clip(w_dc[idx_dc] * scale, WMIN_INH, lat_dc_wmax[idx_dc])
                changed_dc[idx_dc] = True
    if changed_cc.any():
        S_lat_cc.w_inh = w_cc
    if changed_dc.any():
        S_lat_dc.w_inh = w_dc

    # ── Inhibitory Duo ────────────────────────────────────────────────────────
    w_dd = np.array(S_lat_dd.w_inh)
    w_cd = np.array(S_lat_cd.w_inh)
    if w_dd.max() > 0 or w_cd.max() > 0:   # fast-path: skip if all zero
        changed_dd = np.zeros(len(w_dd), dtype=bool)
        changed_cd = np.zeros(len(w_cd), dtype=bool)
        for j in range(N_DUO):
            idx_dd = _ord_dd[_bnd_dd[j]:_bnd_dd[j + 1]]
            idx_cd = _ord_cd[_bnd_cd[j]:_bnd_cd[j + 1]]
            wsum = (w_dd[idx_dd].sum() if len(idx_dd) > 0 else 0.0) + \
                   (w_cd[idx_cd].sum() if len(idx_cd) > 0 else 0.0)
            if wsum > NORM_LIMIT_INH:
                scale = NORM_LIMIT_INH / wsum
                if len(idx_dd) > 0:
                    w_dd[idx_dd] = np.clip(w_dd[idx_dd] * scale, WMIN_INH, lat_dd_wmax[idx_dd])
                    changed_dd[idx_dd] = True
                if len(idx_cd) > 0:
                    w_cd[idx_cd] = np.clip(w_cd[idx_cd] * scale, WMIN_INH, lat_cd_wmax[idx_cd])
                    changed_cd[idx_cd] = True
        if changed_dd.any():
            S_lat_dd.w_inh = w_dd
        if changed_cd.any():
            S_lat_cd.w_inh = w_cd


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
        # Only pass small matrices per-sample; lat_dd and in->duo are too large
        recorder.after_sample(
            sample_idx,
            duration_s,
            w_matrices={
                "in->cnn": _to_matrix(cnn_src_ih, cnn_tgt_ih, w_in_cnn, N_IN, N_CNN),
                "lat_cc":  _to_matrix(lat_cc_src, lat_cc_tgt, w_lat_cc, N_CNN, N_CNN),
            }
        )

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(
        epoch_idx,
        save_dir=SAVE_DIR,
        final_weights={
            "in->cnn": _to_matrix(cnn_src_ih, cnn_tgt_ih, w_in_cnn, N_IN, N_CNN),
            "lat_cc":  _to_matrix(lat_cc_src, lat_cc_tgt, w_lat_cc, N_CNN, N_CNN),
            "lat_cd":  _to_matrix(lat_cd_src, lat_cd_tgt, w_lat_cd, N_CNN, N_DUO),
            "lat_dc":  _to_matrix(lat_dc_src, lat_dc_tgt, w_lat_dc, N_DUO, N_CNN),
        },
        init_weights=None,
    )


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
