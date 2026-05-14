import numpy as np
np.random.seed(42)
from brian2 import *
import glob
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
# TODO: set the path
wav_files = sorted(glob.glob("datasets/vox1_single_person_nano_2/dev/*.wav", recursive=True))
print(f"Found {len(wav_files)} wav files")

EPOCHS   = 8
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Hyperparameters
# ============================================================

N_IN  = 700
N_OUT = 10    # output group

DT_SIM = 1 * ms

# -- Input layer (adaptive LIF) --
tau_m       = 40 * ms
tau_a       = 80 * ms
tau_current = 1 * ms
beta        = 0.8
v_th_in     = 1.0

# -- Output layer (adaptive-threshold LIF) --
tau_h    = 50 * ms
tau_vth  = 100 * ms
vth_rest = 0.8
vth_init = 0.8
vth_jump = 0.3

# -- STDP --
taupre      = 20 * ms
taupost     = 20 * ms
Apre_delta  =  0.004
Apost_delta = -0.0048

# -- Synaptic weight bounds --
wmax = 1.0
wmin = 0.0

# -- Adaptive normalization --
N_PRE_CONNECTED = 140         # fan-in per output neuron (inputs 140-279)
a_pre_inc       = 1.0 / N_PRE_CONNECTED
tau_pre_norm    = 2.0 * second
tau_post_norm   = 2.0 * second
alpha_norm      = 1.0
C_base          = 70        # ~140 synapses * 0.5 mean initial weight
epsilon_norm    = 1e-6

# -- Diagonal/off-diagonal synapse structure --
N_DIAG_PER_OUT      = N_PRE_CONNECTED // N_OUT   # = 14 inputs per diagonal block
wmax_high           = 1.0     # diagonal weight ceiling
wmax_offdiag_factor = 0.3     # off-diagonal ceiling = wmax_high * factor
lr_decay            = 0.5     # lr_factor = lr_decay ^ distance
                              #   diagonal: 1.0 | dist-1: 0.5 | dist-2: 0.25 ...


# ============================================================
# Weight initialisation
# Sparse (N_IN, N_OUT) matrix — only rows 140-279 are active.
# ============================================================

w_io = np.zeros((N_IN, N_OUT))
raw = np.random.uniform(0.3, 0.7, size=(140, N_OUT))
raw /= raw.sum(axis=0)   # normalize each column so sum = 1 (matches C_base = 1)
w_io[140:280, :] = raw

init_weights = {"in->out": w_io.copy()}


# ============================================================
# Build Brian2 network
# ============================================================

defaultclock.dt = DT_SIM

# ── Input neurons ──────────────────────────────────────────────────────────────
_dummy_I = np.zeros((1, N_IN), dtype=float)
I_timed  = TimedArray(_dummy_I, dt=DT_SIM)

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

# ── Output neurons ─────────────────────────────────────────────────────────────
#
# xpre  — slow leaky trace of incoming presynaptic activity.
#          Incremented in on_pre by a_pre_inc = 1/n_j (normalised fan-in).
# xpost — slow leaky trace of postsynaptic firing.
#          Incremented by 1 on each spike.
# Both decay exponentially with tau_pre_norm / tau_post_norm (1-5 s).
# They are saved to numpy after each sample and restored after net.restore()
# so they accumulate correctly across samples despite the restore call.
#
eqs_out = f"""
dv/dt      = -v / tau_h                              : 1
dvth/dt    = -(vth - {vth_rest}) / tau_vth           : 1
dxpre/dt  = -xpre  / tau_pre_norm                   : 1
dxpost/dt = -xpost / tau_post_norm                  : 1
"""
G_out = NeuronGroup(
    N_OUT, eqs_out,
    threshold="v > vth",
    reset=f"v=0; vth=vth+{vth_jump}; xpost+=1",
    refractory=2 * ms,
    method="euler"
)
G_out.vth = vth_init

# ── STDP synapses: input → output ─────────────────────────────────────────────
stdp_model = """
w          : 1
wmax_syn   : 1
lr_syn     : 1
dapre/dt   = -apre  / taupre  : 1 (event-driven)
dapost/dt  = -apost / taupost : 1 (event-driven)
"""

on_pre  = ("v_post += w\n"
           "apre += lr_syn * Apre_delta\n"
           "w = clip(w + apost*(w-wmin), wmin, wmax_syn)\n"
           "xpre_post += a_pre_inc")

on_post = ("apost += lr_syn * Apost_delta\n"
           "w = clip(w + apre*(wmax_syn-w), wmin, wmax_syn)")

S = Synapses(G_in, G_out, model=stdp_model, on_pre=on_pre, on_post=on_post)
S.connect(i=list(range(140, 280)) * N_OUT,
          j=[j for j in range(N_OUT) for _ in range(140)])

# inject a_pre_inc as a namespace constant (not a Brian2 parameter)
S.namespace['a_pre_inc'] = a_pre_inc

src_io = np.array(S.i)
tgt_io = np.array(S.j)

# Per-synapse diagonal distance and structured wmax / lr
group_io = (src_io - 140) // N_DIAG_PER_OUT                          # 0–9
dist_io  = np.abs(group_io.astype(int) - tgt_io.astype(int))         # 0–9
wmax_io  = np.where(dist_io == 0, wmax_high, wmax_high * wmax_offdiag_factor)
lr_io    = lr_decay ** dist_io.astype(float)                          # 1.0 → 0.002
S.wmax_syn = wmax_io
S.lr_syn   = lr_io

print(f"Synapses created: {len(src_io)}")

# ── Pre-compute per-output-neuron synapse index masks ─────────────────────────
tgt_masks = [np.where(tgt_io == j)[0] for j in range(N_OUT)]

# ── Spike monitor for normalization (tracks cumulative count) ─────────────────
spike_mon_out = SpikeMonitor(G_out)

# ── WTA network_operation ─────────────────────────────────────────────────────
# Runs after state integration but before threshold checking.
# Identifies candidates (v >= vth) and keeps only the most excited one.
@network_operation(when='before_thresholds')
def wta_op():
    v   = np.array(G_out.v)
    vth = np.array(G_out.vth)
    above = v >= vth
    n_above = int(above.sum())
    if n_above <= 1:
        return                          # 0 or 1 candidate, no competition
    candidates = np.where(above)[0]
    diffs = v[candidates] - vth[candidates]
    winner = candidates[np.argmax(diffs)]
    suppress = candidates[candidates != winner]
    G_out.v[suppress] = 0.0


# ── Adaptive normalization network_operation ──────────────────────────────────
# Runs at end of each timestep; applies normalization only for neurons that
# just spiked. Preserves relative STDP weight structure, anchors total to C_j.
_prev_spike_count = np.zeros(N_OUT, dtype=np.int64)

@network_operation(when='end')
def normalize_on_spike():
    curr = np.array(spike_mon_out.count)
    just_spiked = np.where(curr > _prev_spike_count)[0]
    _prev_spike_count[:] = curr
    if len(just_spiked) == 0:
        return

    a_pre  = np.array(G_out.xpre)
    a_post = np.array(G_out.xpost)
    R = (a_pre - a_post) / (a_pre + a_post + epsilon_norm)   # R ∈ (-1, 1); zero at xpre == xpost
    C = C_base * (1.0 + alpha_norm * R)                      # C ∈ (0, 2·C_base)

    for j in just_spiked:
        idx = tgt_masks[j]
        w_j    = np.array(S.w[idx])
        wmax_j = np.array(S.wmax_syn[idx])
        total  = w_j.sum() + epsilon_norm
        S.w[idx] = np.clip(C[j] * w_j / total, wmin, wmax_j)


net = Network(G_in, G_out, S, spike_mon_out, wta_op, normalize_on_spike)


# ============================================================
# Recorder setup
# ============================================================

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record_and_visualize_config.yaml")
recorder = Recorder(CONFIG_PATH, net)

recorder.track_group("input",  G_in)
recorder.track_group("output", G_out)

recorder.track_synapses("in->out", S, src_io, tgt_io)

recorder.build()

G_out.vth = vth_init
net.store('init')


# ============================================================
# Cross-sample activity traces (persist despite net.restore)
# ============================================================

xpre_saved  = np.zeros(N_OUT)
xpost_saved = np.zeros(N_OUT)


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

        # ── Encode audio ───────────────────────────────────────────────────────
        try:
            I, T = compute_spike_input_current(
                audio_path,
                scale=1,
                sustained_per_band=4,
                onset_per_band=2,
                phase_per_band=1,
                sust_spread_min=0.7,
                sust_spread_max=1.3,
            )
        except Exception as e:
            print(f"    Error encoding audio: {e}")
            continue

        duration_s = float(T) * float(DT_SIM)

        # ── Reset to clean state ──────────────────────────────────────────────
        net.restore('init')

        G_in.namespace["I_timed"] = TimedArray(I.T.astype(float), dt=DT_SIM)

        # Weights survive across samples; STDP traces reset cleanly.
        S.w     = w_io[src_io, tgt_io]
        S.apre  = 0
        S.apost = 0

        # Restore slow activity traces (not part of net.store snapshot).
        G_out.xpre  = xpre_saved
        G_out.xpost = xpost_saved

        # Spike monitor count resets with net.restore; re-sync the tracker.
        _prev_spike_count[:] = 0

        recorder._elapsed_ms = 0.0

        # ── Record: before ─────────────────────────────────────────────────────
        recorder.before_sample(sample_idx)

        # ── Simulate ───────────────────────────────────────────────────────────
        net.run(T * DT_SIM)

        # ── Save activity traces for next sample ───────────────────────────────
        xpre_saved  = np.array(G_out.xpre)
        xpost_saved = np.array(G_out.xpost)

        # ── Extract updated weights ────────────────────────────────────────────
        w_io_new = np.zeros((N_IN, N_OUT))
        w_io_new[src_io, tgt_io] = np.array(S.w)
        w_io = w_io_new

        # ── Record: after ──────────────────────────────────────────────────────
        recorder.after_sample(
            sample_idx,
            duration_s,
            w_matrices={"in->out": w_io}
        )

    # ── End of epoch ──────────────────────────────────────────────────────────
    recorder.save_epoch(
        epoch_idx,
        save_dir=SAVE_DIR,
        final_weights={"in->out": w_io},
        init_weights=init_weights if epoch_idx == 0 else None,
    )


# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Training complete — {time.time() - start:.2f}s")
print(f"{'='*60}")
