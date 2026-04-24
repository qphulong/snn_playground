"""
train.py
--------
Orchestration script.  All architecture knowledge lives in
architecture.yaml; all recording knowledge lives in record_config.yaml.
This file is responsible only for the training loop.

Flow
----
  1. Load architecture + record config
  2. For each epoch:
       For each audio sample:
         a. Encode audio → I_arr (T × N_input)
         b. start_scope()  [Brian2 reset]
         c. build_network()  →  net
         d. inject_weights()  [carry weights across samples]
         e. recorder.setup_sample()  →  ops  [attach monitors]
         f. brian2 run()
         g. recorder.collect_sample()  [harvest + normalise]
         h. (weights are now updated in net.weights)
       recorder.save_epoch()
       recorder.reset_epoch()
  3. Save initial and final weight matrices
  4. recorder.log_top_k()
"""

import glob
import os
import sys
import time

import numpy as np
from brian2 import (
    Network,
    defaultclock,
    ms,
    start_scope,
)

np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.spike_encoding import compute_spike_input_current

from network_builder import (
    load_architecture,
    build_network,
    extract_weights,
    inject_weights,
)
from recorder import load_record_config, Recorder

# ============================================================
# Paths
# ============================================================

HERE          = os.path.dirname(os.path.abspath(__file__))
ARCH_PATH     = os.path.join(HERE, "architecture.yaml")
REC_CFG_PATH  = os.path.join(HERE, "record_config.yaml")
OUT_DIR       = HERE

# ============================================================
# Load configs
# ============================================================

arch    = load_architecture(ARCH_PATH)
rec_cfg = load_record_config(REC_CFG_PATH)

EPOCHS = int(rec_cfg.get("epochs", 1))
DT_MS  = float(arch.get("simulation", {}).get("dt_ms", 1.0))
DT_SIM = DT_MS * ms

# ============================================================
# Dataset
# ============================================================

wav_files = sorted(
    glob.glob("datasets/vox1_nano_abc/dev/**/*.wav", recursive=True)
)
print(f"Found {len(wav_files)} files in datasets/vox1_nano/dev/")

# ============================================================
# Identify input group(s) from architecture
# ============================================================

input_groups = [
    name
    for name, cfg in arch["neuron_groups"].items()
    if cfg.get("is_input", False)
]
if not input_groups:
    raise RuntimeError("No neuron group has 'is_input: true' in architecture.yaml")

# For now we assume a single input group drives the TimedArray
input_group_name = input_groups[0]
N_input = int(arch["neuron_groups"][input_group_name]["n"])

# ============================================================
# Initialise weights (carried across samples within an epoch,
# and across epochs — learning is cumulative)
# ============================================================

# We do a dry build with a dummy input to get initial weight matrices
_dummy_I = np.zeros((1, N_input))
_dummy_net = build_network(arch, _dummy_I)
w_current = extract_weights(_dummy_net)   # dict[synapse_name → ndarray]

# Save initial weights
init_save_path = os.path.join(OUT_DIR, "history_init.npz")
np.savez_compressed(
    init_save_path,
    **{f"w_{sname}_init": W.astype(np.float32) for sname, W in w_current.items()},
)
print(f"Saved initial weights → {init_save_path}")

# ============================================================
# Training loop
# ============================================================

recorder = Recorder(rec_cfg, arch)
start    = time.time()

for epoch_idx in range(EPOCHS):
    print(f"\n{'='*60}\nEpoch {epoch_idx}/{EPOCHS - 1}\n{'='*60}")
    recorder.reset_epoch()

    for sample_idx, audio_path in enumerate(wav_files):
        print(
            f"[epoch {epoch_idx}/{EPOCHS-1}, "
            f"sample {sample_idx}/{len(wav_files)-1}]  "
            f"{os.path.relpath(audio_path)}"
        )

        # ── 1. Encode audio ───────────────────────────────────────────────────
        try:
            I_arr, T = compute_spike_input_current(
                audio_path,
                scale=1,
                sustained_per_band=4,
                onset_per_band=2,
                phase_per_band=1,
                sust_spread_min=0.7,
                sust_spread_max=1.3,
            )
        except Exception as e:
            print(f"  Error processing audio: {e}")
            continue

        # I_arr: shape (T, N_input) — T simulation timesteps at DT_MS per step.
        # duration_s must reflect the real audio duration, not T * DT_MS,
        # because the encoder maps N audio samples -> T << N timesteps
        # (e.g. one timestep per spectrogram frame, not per audio sample).
        import librosa
        _audio_dur_s  = librosa.get_duration(path=audio_path)
        duration_s    = _audio_dur_s
        sim_duration_s = float(T) * DT_MS * 1e-3   # actual simulation wall-time

        # ── 2. Build network ──────────────────────────────────────────────────
        start_scope()
        defaultclock.dt = DT_SIM

        net = build_network(arch, I_arr.T)  # build_network expects (T, N)

        # ── 3. Carry weights from previous sample ─────────────────────────────
        inject_weights(net, w_current)

        # ── 4. Attach monitors ────────────────────────────────────────────────
        ops = recorder.setup_sample(net, dt_ms=DT_MS)

        # ── 5. Run ────────────────────────────────────────────────────────────
        brian_net = Network(
            *net.groups.values(),
            *net.synapses.values(),
            *ops.spike_monitors.values(),
            *ops.vmon.values(),
            *ops.net_ops,
        )
        brian_net.run(T * DT_SIM)

        # ── 6. Collect + normalise ────────────────────────────────────────────
        recorder.collect_sample(
            net, ops,
            T_steps        = T,
            dt_ms          = DT_MS,
            duration_s     = duration_s,
            sim_duration_s = sim_duration_s,
        )

        # ── 7. Update carried weights ─────────────────────────────────────────
        w_current = dict(net.weights)   # normalised matrices from collect_sample

    # ── Save epoch ────────────────────────────────────────────────────────────
    recorder.save_epoch(epoch_idx, OUT_DIR, w_current)

# ============================================================
# Post-training
# ============================================================

print(f"\n{'='*60}\nTraining Complete!\n{'='*60}")

# Save final weight matrices
final_save_path = os.path.join(OUT_DIR, "history_final.npz")
np.savez_compressed(
    final_save_path,
    **{f"w_{sname}_final": W.astype(np.float32) for sname, W in w_current.items()},
)
print(f"Saved final weights → {final_save_path}")

# Log top-k weights
recorder.log_top_k(w_current, OUT_DIR)

print(f"\nTotal runtime: {time.time() - start:.2f} s")