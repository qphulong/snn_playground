"""
WeightHistoryStore — per-sample weight matrix history.

Enabled when training config has `save_weights_history: true`.
Stores each synapse's full weight matrix after every sample in an HDF5 file,
compressed with gzip. This enables the UI to show how any specific weight
w[src_idx, tgt_idx] evolves over training.

File layout:
    weight_history.h5
    └── <synapse_name>/
        ├── sample_000000   (N_src × N_tgt float32)
        ├── sample_000001
        └── ...
"""

import os
import numpy as np
import h5py
from typing import Dict, List, Tuple


class WeightHistoryStore:

    CHUNK_ROWS = 100
    CHUNK_COLS = 100

    def __init__(self, filepath: str):
        self.filepath = filepath

    # ------------------------------------------------------------------

    def append(self, sample_idx: int, weights: Dict[str, np.ndarray]):
        """Append weight snapshot for one training sample."""
        with h5py.File(self.filepath, 'a') as f:
            for syn_name, w in weights.items():
                grp = f.require_group(syn_name)
                ds_name = f"sample_{sample_idx:06d}"
                if ds_name in grp:
                    del grp[ds_name]   # overwrite if resuming
                chunk_r = min(self.CHUNK_ROWS, w.shape[0])
                chunk_c = min(self.CHUNK_COLS, w.shape[1])
                grp.create_dataset(
                    ds_name,
                    data=w.astype(np.float32),
                    chunks=(chunk_r, chunk_c),
                    compression='gzip',
                    compression_opts=4,
                )

    # ------------------------------------------------------------------

    def get_weight_evolution(
        self,
        syn_name: str,
        src_idx: int,
        tgt_idx: int,
    ) -> Tuple[List[int], List[float]]:
        """
        Return (sample_indices, weight_values) for a specific synapse weight.

        Args:
            syn_name: name of the synapse (e.g. 'input_to_hidden')
            src_idx:  source neuron index
            tgt_idx:  target neuron index
        """
        if not os.path.exists(self.filepath):
            return [], []

        sample_indices, values = [], []
        with h5py.File(self.filepath, 'r') as f:
            if syn_name not in f:
                return [], []
            grp = f[syn_name]
            for ds_name in sorted(grp.keys()):
                idx = int(ds_name.split('_')[1])
                val = float(grp[ds_name][src_idx, tgt_idx])
                sample_indices.append(idx)
                values.append(val)

        return sample_indices, values

    def list_synapses(self) -> List[str]:
        if not os.path.exists(self.filepath):
            return []
        with h5py.File(self.filepath, 'r') as f:
            return list(f.keys())

    def num_samples(self, syn_name: str) -> int:
        if not os.path.exists(self.filepath):
            return 0
        with h5py.File(self.filepath, 'r') as f:
            if syn_name not in f:
                return 0
            return len(f[syn_name])
