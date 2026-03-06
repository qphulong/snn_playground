"""
Trainer — training loop with checkpointing.

Works with any SNNNetwork architecture. The config drives everything;
no hyperparameters are hardcoded here.
"""

import os
import time
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional

from .network import SNNNetwork
from .audio_encoding import compute_input_current
from .dataset import Dataset
from storage.weight_history_store import WeightHistoryStore


class Trainer:
    """
    Runs the training loop over an audio dataset.

    Each iteration:
        1. Load audio → compute input currents
        2. Run Brian2 simulation
        3. Extract updated weights from all synapses
        4. Apply homeostatic normalization (per learning rule)
        5. Record diagnostics
        6. Save checkpoint periodically
    """

    def __init__(
        self,
        network: SNNNetwork,
        config: Any,
        checkpoint_manager: Any,
        weight_history_store: Optional[WeightHistoryStore] = None,
    ):
        self.network = network
        self.config = config           # ExperimentConfig
        self.checkpoint_mgr = checkpoint_manager
        self.weight_history_store = weight_history_store

    # ------------------------------------------------------------------

    def train_one_sample(
        self,
        audio_path: str,
        weights: Dict[str, np.ndarray],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Run one training step on a single audio file.

        Returns:
            (updated_weights, diagnostics_dict)
        """
        I, T = compute_input_current(audio_path, self.config.audio)

        output = self.network.run_simulation(
            input_currents=I,
            weights=weights,
            dt_ms=self.config.training.dt,
        )

        w_prev = {k: v.copy() for k, v in weights.items()}
        w_new  = output['weights']

        # Apply homeostatic normalization per synapse learning rule
        for syn_name, syn_info in self.network.synapses.items():
            lr = syn_info['learning_rule']
            w_new[syn_name] = lr.apply_homeostasis(w_new[syn_name])

        # Collect diagnostics (aggregate across all synapses)
        diag = {}
        for syn_name, w in w_new.items():
            prev = w_prev.get(syn_name, np.zeros_like(w))
            diag[f'{syn_name}/mean_w']   = float(w.mean())
            diag[f'{syn_name}/std_w']    = float(w.std())
            diag[f'{syn_name}/delta_w']  = float(np.abs(w - prev).mean())

        for layer_name, spike_data in output['spikes'].items():
            diag[f'{layer_name}/num_spikes'] = spike_data['num_spikes']

        return w_new, diag

    # ------------------------------------------------------------------

    def run_training(
        self,
        dataset: Dataset,
        resume_from: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            dataset:      Audio dataset (iterable of file paths)
            resume_from:  Checkpoint ID to resume from (or None)

        Returns:
            history dict — {metric_name: [values per sample]}
        """
        # ── Init or restore ──────────────────────────────────────────
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
            ckpt     = self.checkpoint_mgr.load(resume_from)
            weights  = ckpt['weights']
            start    = ckpt['sample_idx'] + 1
            history  = ckpt['history']
        else:
            weights  = self.network.init_weights()
            start    = 0
            history  = defaultdict(list)

        max_samples = getattr(self.config.training, 'max_samples', len(dataset))
        end = min(start + max_samples, len(dataset)) if start == 0 else min(len(dataset), start + max_samples)

        print(f"Training: samples {start} → {end - 1}  |  experiment: {self.config.name}")

        # ── Loop ─────────────────────────────────────────────────────
        for idx in range(start, end):
            audio_path = dataset[idx]
            sample_num = idx - start

            if sample_num % 5 == 0:
                print(f"  [{idx:4d}] {os.path.relpath(audio_path)}")

            try:
                weights, diag = self.train_one_sample(audio_path, weights)
            except Exception as exc:
                print(f"  WARNING: skipped sample {idx} — {exc}")
                continue

            for key, val in diag.items():
                history[key].append(val)

            # ── Weight history ───────────────────────────────────────
            if self.weight_history_store is not None:
                self.weight_history_store.append(idx, weights)

            # ── Checkpoint ───────────────────────────────────────────
            interval = getattr(self.config.training, 'checkpoint_interval', 5)
            if (idx + 1) % interval == 0:
                self.checkpoint_mgr.save({
                    'weights':    weights,
                    'sample_idx': idx,
                    'history':    dict(history),
                    'config':     self.config,
                    'timestamp':  time.time(),
                })

        # ── Final checkpoint ─────────────────────────────────────────
        self.checkpoint_mgr.save({
            'weights':    weights,
            'sample_idx': end - 1,
            'history':    dict(history),
            'config':     self.config,
            'timestamp':  time.time(),
        })

        print(f"\nTraining complete. {end - start} samples processed.")
        return dict(history)
