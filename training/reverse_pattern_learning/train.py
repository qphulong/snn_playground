"""
train.py  -  SNN AB/BA pattern discrimination
============================================================================
Run:   python3 training/reverse_pattern_learning/train.py

This script demonstrates a simple network with 2 layers, each containing 2 neurons,
fully connected. The goal is for each neuron to learn the AB and BA patterns.

The history of the training process will be saved to a file:
training/reverse_pattern_learning/snn_history.pkl

Visualization:
- The training history can be visualized using: 
  python3 training/reverse_pattern_learning/visualize.py
  This will show the spiking pattern, how weights evolve, and how the membrane
  potential changes within each neuron over time.

Naming rules to avoid Brian2 namespace conflicts:
- No variable ending in _pre or _post anywhere.
- Function arguments or locals that hold weights or thresholds are named `init_weights` / `init_thresh`
  (never `w` or `theta` in the `run_sample` scope).
- All numeric hyperparameters are baked into equation strings via f-strings, so nothing leaks
  from Python scope into Brian2's run namespace.
"""

import os
import numpy as np
import pickle
from brian2 import *

random_seed = np.random.randint(0, 1_000_000)
np.random.seed(random_seed)
defaultclock.dt = 0.01 * ms

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join('training', 'reverse_pattern_learning')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================

SAMPLE_DUR      = 10.0   # ms
SPIKE_OFFSET_1  =  4.0   # ms  first spike in pattern
SPIKE_OFFSET_2  =  6.0   # ms  second spike

N_AB = 100
N_BA = 100

# LIF / membrane
TAU_M         =  2.0   # ms
V_REST        =  0.0
V_RESET_VAL   =  0.0   # refractory clamp potential
V_THRESH_INIT =  0.60   # initial adaptive threshold
TAU_THRESH    =  10   # ms  threshold decay
THRESH_JUMP   =  0.25  # threshold increment on spike
THRESH_REST   =  0.60  # resting threshold level
REFRAC_MS     =  2.0   # ms

# Lateral inhibition
LAT_INH       =  0.5

# STDP
TAU_SYN_TRACE = 20.0   # ms  pre-synaptic eligibility trace
TAU_NRN_TRACE = 20.0   # ms  post-synaptic eligibility trace
A_PLUS        =  0.035
A_MINUS       =  0.025
SYN_MIN       =  0.0
SYN_MAX       =  1.0
SYN_INIT      =  0.75

# Weight normalisation
NORM_MARGIN   =  1.15

# Refractory PSP scaling  (1 - 0.95 = 0.05)
REFRAC_SCALE  =  0.01

# ============================================================
# STIMULUS SEQUENCE
# ============================================================

labels   = ['AB'] * N_AB + ['BA'] * N_BA
rng      = np.random.default_rng(42)
sequence = [labels[i] for i in rng.permutation(len(labels))]

print(f"Sequence: {len(sequence)} samples  "
      f"({sequence.count('AB')} AB,  {sequence.count('BA')} BA)")

# ============================================================
# PER-SAMPLE SIMULATION
# ============================================================

def run_sample(pattern, init_weights, init_thresh):
    """
    Simulate one 10-ms sample.

    Parameters
    ----------
    pattern      : 'AB' or 'BA'
    init_weights : np.ndarray  [w_CA, w_CB, w_DA, w_DB]  – NOT named 'w'
    init_thresh  : np.ndarray  [thresh_C, thresh_D]       – NOT named 'theta'

    Returns
    -------
    new_weights, new_thresh, traces
    """
    start_scope()

    # ── input spikes ────────────────────────────────────────
    if pattern == 'AB':
        t_A = SPIKE_OFFSET_1 * ms
        t_B = SPIKE_OFFSET_2 * ms
    else:
        t_A = SPIKE_OFFSET_2 * ms
        t_B = SPIKE_OFFSET_1 * ms

    inp = SpikeGeneratorGroup(2, indices=[0, 1], times=[t_A, t_B], name='inp')

    # ── output layer  (adaptive-threshold LIF) ───────────────
    # All constants baked into the string – nothing leaks from Python scope.
    # Variable names:
    #   vth         : adaptive threshold  (avoids 'theta' conflict)
    #   eligibility : post-synaptic STDP trace
    #   refrac_flag : refractory boolean flag
    eqs_out = (
        f'dv/dt           = (-(v - {V_REST})) / ({TAU_M}*ms)           : 1 (unless refractory)\n'
        f'dvth/dt         = -(vth - {THRESH_REST}) / ({TAU_THRESH}*ms) : 1\n'
        f'deligibility/dt = -eligibility / ({TAU_NRN_TRACE}*ms)        : 1\n'
        f'refrac_flag     : boolean\n'
    )

    reset_out = (
        f'v           = {V_RESET_VAL}\n'
        f'vth         = vth + {THRESH_JUMP}\n'
        f'eligibility = eligibility + 1.0\n'
        f'refrac_flag = True\n'
    )

    out = NeuronGroup(
        2,
        model      = eqs_out,
        threshold  = 'v > vth',
        reset      = reset_out,
        refractory = f'{REFRAC_MS}*ms',
        method     = 'euler',
        name       = 'out',
    )
    out.v           = V_REST
    out.vth         = init_thresh          # safe: 'init_thresh' not internal
    out.eligibility = 0.0
    out.refrac_flag = False

    # ── feedforward synapses + STDP ──────────────────────────
    # Synapse variable: syn_wt  (avoids 'w' conflict)
    #                   syn_eligibility  (pre-synaptic trace)
    #
    # Brian2 connect() order for 2->2 full:
    #   (A->C)=0, (A->D)=1, (B->C)=2, (B->D)=3
    #
    # Canonical weight vector  [w_CA, w_CB, w_DA, w_DB]:
    #   w_CA  A->C  idx 0
    #   w_CB  B->C  idx 2
    #   w_DA  A->D  idx 1
    #   w_DB  B->D  idx 3
    #
    # Brian2 auto-generates _post accessors inside synapse rules:
    #   eligibility_post   ->  eligibility of the post neuron
    #   refrac_flag_post   ->  refrac_flag of the post neuron

    syn_eqs = (
        f'syn_wt              : 1\n'
        f'dsyn_eligibility/dt = -syn_eligibility / ({TAU_SYN_TRACE}*ms) : 1 (event-driven)\n'
    )

    syn_on_pre = (
        f'syn_eligibility += 1.0\n'
        f'syn_wt           = clip(syn_wt - {A_MINUS}*eligibility_post, {SYN_MIN}, {SYN_MAX})\n'
        f'v_post          += syn_wt * (refrac_flag_post * {REFRAC_SCALE} + (1 - refrac_flag_post) * 1.0)\n'
    )

    syn_on_post = (
        f'syn_wt = clip(syn_wt + {A_PLUS}*syn_eligibility, {SYN_MIN}, {SYN_MAX})\n'
    )

    ff = Synapses(inp, out,
                  model   = syn_eqs,
                  on_pre  = syn_on_pre,
                  on_post = syn_on_post,
                  name    = 'ff')
    ff.connect()

    # load into Brian2 index order
    loaded         = np.zeros(4)
    loaded[0]      = init_weights[0]   # w_CA -> A->C -> idx 0
    loaded[1]      = init_weights[2]   # w_DA -> A->D -> idx 1
    loaded[2]      = init_weights[1]   # w_CB -> B->C -> idx 2
    loaded[3]      = init_weights[3]   # w_DB -> B->D -> idx 3
    ff.syn_wt          = loaded
    ff.syn_eligibility = 0.0

    # ── lateral inhibition ───────────────────────────────────
    lat = Synapses(out, out, on_pre=f'v_post = clip(v_post - {LAT_INH}, 0, inf)', name='lat')
    lat.connect(condition='i != j')

    # ── monitors ─────────────────────────────────────────────
    st_mon  = StateMonitor(out, ['v', 'vth'], record=True, name='st_mon')
    spk_out = SpikeMonitor(out, name='spk_out')
    spk_inp = SpikeMonitor(inp, name='spk_inp')

    run(SAMPLE_DUR * ms)

    # ── unpack weights to canonical order ────────────────────
    raw         = np.array(ff.syn_wt)
    new_weights = np.clip([raw[0], raw[2], raw[1], raw[3]], SYN_MIN, SYN_MAX)
    new_thresh  = np.array(out.vth)

    # ── weight normalisation ─────────────────────────────────
    for nrn in range(2):
        if nrn in spk_out.i:
            idxs  = [0, 1] if nrn == 0 else [2, 3]
            limit = new_thresh[nrn] * NORM_MARGIN
            wsum  = new_weights[idxs].sum()
            if wsum > limit > 0:
                new_weights[idxs] *= limit / wsum
    new_weights = np.clip(new_weights, SYN_MIN, SYN_MAX)

    traces = dict(
        t       = np.array(st_mon.t / ms),
        vm_C    = np.array(st_mon.v[0]),
        vm_D    = np.array(st_mon.v[1]),
        theta_C = np.array(st_mon.vth[0]),
        theta_D = np.array(st_mon.vth[1]),
        spk_C   = np.array(spk_out.t[spk_out.i == 0] / ms),
        spk_D   = np.array(spk_out.t[spk_out.i == 1] / ms),
        spk_A   = np.array(spk_inp.t[spk_inp.i  == 0] / ms),
        spk_B   = np.array(spk_inp.t[spk_inp.i  == 1] / ms),
    )

    return new_weights, new_thresh, traces


# ============================================================
# TRAINING LOOP
# ============================================================

cur_weights = np.array([0.56, 0.72, 0.81, 0.49])
cur_thresh  = np.full(2, V_THRESH_INIT)

history = dict(
    sequence = sequence,
    weights  = [cur_weights.copy()],   # index 0 = before any training
    vm_C=[], vm_D=[],
    theta_C=[], theta_D=[],
    t_trace=[],
    spikes_C=[], spikes_D=[],
    spikes_A=[], spikes_B=[],
)

for idx, pattern in enumerate(sequence):
    print(f"  [{idx+1:3d}/{len(sequence)}]  {pattern}  "
          f"weights=[{', '.join(f'{v:.3f}' for v in cur_weights)}]")

    cur_weights, cur_thresh, tr = run_sample(pattern, cur_weights, cur_thresh)

    history['weights'].append(cur_weights.copy())
    history['vm_C'].append(tr['vm_C'])
    history['vm_D'].append(tr['vm_D'])
    history['theta_C'].append(tr['theta_C'])
    history['theta_D'].append(tr['theta_D'])
    history['t_trace'].append(tr['t'])
    history['spikes_C'].append(tr['spk_C'])
    history['spikes_D'].append(tr['spk_D'])
    history['spikes_A'].append(tr['spk_A'])
    history['spikes_B'].append(tr['spk_B'])

pkl_path = os.path.join(OUT_DIR, 'snn_history.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(history, f)

print(f'\nTraining complete.  Saved -> {pkl_path}')
print(f'Final weights  w_CA={cur_weights[0]:.4f}  w_CB={cur_weights[1]:.4f}  '
      f'w_DA={cur_weights[2]:.4f}  w_DB={cur_weights[3]:.4f}')