"""
ModelStorage — chunked HDF5 storage for large weight matrices.

Enables lazy loading: the web API fetches only requested (row, col) regions
without loading the full matrix into memory. Critical for large graph networks.
"""

import os
import numpy as np
import h5py
from typing import Dict, Optional


class ModelStorage:
    """HDF5-backed weight matrix store with region-based access."""

    # Chunk size — trades off between random-access latency and total I/O.
    # (100, 100) means each read loads at most 100×100 weights.
    CHUNK_ROWS = 100
    CHUNK_COLS = 100

    def __init__(self, filepath: str):
        self.filepath = filepath

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_weights(self, weights: Dict[str, np.ndarray]):
        """
        Save all synapse weight matrices with chunking.

        Args:
            weights: {syn_name: (N_src, N_tgt) float32 array}
        """
        with h5py.File(self.filepath, 'w') as f:
            grp = f.require_group('weights')
            for syn_name, w in weights.items():
                chunk_r = min(self.CHUNK_ROWS, w.shape[0])
                chunk_c = min(self.CHUNK_COLS, w.shape[1])
                ds = grp.create_dataset(
                    syn_name,
                    data=w.astype(np.float32),
                    chunks=(chunk_r, chunk_c),
                    compression='gzip',
                    compression_opts=4,
                )
                ds.attrs['shape'] = w.shape

    # ------------------------------------------------------------------
    # Read — full
    # ------------------------------------------------------------------

    def load_weights(self) -> Dict[str, np.ndarray]:
        """Load all weight matrices."""
        weights = {}
        with h5py.File(self.filepath, 'r') as f:
            for name in f['weights']:
                weights[name] = f['weights'][name][:]
        return weights

    def list_synapses(self):
        with h5py.File(self.filepath, 'r') as f:
            return list(f['weights'].keys())

    def weight_shape(self, syn_name: str):
        with h5py.File(self.filepath, 'r') as f:
            return tuple(f['weights'][syn_name].shape)

    # ------------------------------------------------------------------
    # Read — lazy region
    # ------------------------------------------------------------------

    def load_region(
        self,
        syn_name: str,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> np.ndarray:
        """
        Load a rectangular region from a weight matrix.

        Only reads the requested HDF5 chunks — efficient for large matrices.
        """
        with h5py.File(self.filepath, 'r') as f:
            ds = f['weights'][syn_name]
            total_rows, total_cols = ds.shape
            row_end = min(row_end, total_rows)
            col_end = min(col_end, total_cols)
            return ds[row_start:row_end, col_start:col_end]

    def load_row_stats(self, syn_name: str) -> Dict[str, np.ndarray]:
        """Per-row mean and std (per-source-neuron outgoing weight stats)."""
        with h5py.File(self.filepath, 'r') as f:
            w = f['weights'][syn_name][:]
        return {'mean': w.mean(axis=1), 'std': w.std(axis=1)}

    def load_col_stats(self, syn_name: str) -> Dict[str, np.ndarray]:
        """Per-column mean and std (per-target-neuron incoming weight stats)."""
        with h5py.File(self.filepath, 'r') as f:
            w = f['weights'][syn_name][:]
        return {'mean': w.mean(axis=0), 'std': w.std(axis=0)}
