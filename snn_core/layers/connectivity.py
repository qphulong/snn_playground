from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
import scipy.sparse as sp


class Connectivity(ABC):
    """
    Abstract connectivity pattern.

    Subclasses define which (source, target) neuron pairs are connected.
    Used by SNNNetwork to call Synapses.connect(i=src, j=tgt).
    """

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def get_connections(self, N_src: int, N_tgt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (src_indices, tgt_indices) arrays for Brian2 Synapses.connect().

        Args:
            N_src: number of source neurons
            N_tgt: number of target neurons

        Returns:
            src: 1-D int array of source indices
            tgt: 1-D int array of target indices
        """

    def get_adjacency_matrix(self, N_src: int, N_tgt: int) -> sp.csr_matrix:
        """Return sparse adjacency matrix (optional, used by viz)."""
        src, tgt = self.get_connections(N_src, N_tgt)
        data = np.ones(len(src), dtype=np.float32)
        return sp.csr_matrix((data, (src, tgt)), shape=(N_src, N_tgt))


class DenseConnectivity(Connectivity):
    """All-to-all (fully connected) connectivity."""

    def get_connections(self, N_src: int, N_tgt: int) -> Tuple[np.ndarray, np.ndarray]:
        src = np.repeat(np.arange(N_src), N_tgt)
        tgt = np.tile(np.arange(N_tgt), N_src)
        return src.astype(np.int32), tgt.astype(np.int32)


class SparseConnectivity(Connectivity):
    """
    Randomly sparse connectivity with configurable sparsity.

    Config:
        sparsity (float): fraction of connections to keep (0 < sparsity <= 1)
        seed (int): random seed for reproducibility
    """

    def get_connections(self, N_src: int, N_tgt: int) -> Tuple[np.ndarray, np.ndarray]:
        sparsity = self.config.get('sparsity', 0.1)
        seed = self.config.get('seed', 0)
        rng = np.random.default_rng(seed)

        all_src = np.repeat(np.arange(N_src), N_tgt)
        all_tgt = np.tile(np.arange(N_tgt), N_src)

        n_total = N_src * N_tgt
        n_keep = max(1, int(n_total * sparsity))
        mask = rng.choice(n_total, size=n_keep, replace=False)

        return all_src[mask].astype(np.int32), all_tgt[mask].astype(np.int32)


class GraphConnectivity(Connectivity):
    """
    Graph-based connectivity loaded from a pre-computed adjacency file.

    Config:
        graph_source (str): path to .npz file containing 'src' and 'tgt' arrays,
                            OR a scipy sparse matrix saved via sp.save_npz()
        sparsity (float):   optional — subsample edges if graph is too dense
    """

    def get_connections(self, N_src: int, N_tgt: int) -> Tuple[np.ndarray, np.ndarray]:
        graph_path = self.config.get('graph_source')
        if graph_path is None:
            raise ValueError("GraphConnectivity requires 'graph_source' in config")

        data = np.load(graph_path, allow_pickle=True)

        if 'src' in data and 'tgt' in data:
            src = data['src'].astype(np.int32)
            tgt = data['tgt'].astype(np.int32)
        else:
            # Assume scipy sparse matrix
            adj = sp.load_npz(graph_path)
            adj_coo = adj.tocoo()
            src = adj_coo.row.astype(np.int32)
            tgt = adj_coo.col.astype(np.int32)

        # Optional subsampling
        sparsity = self.config.get('sparsity', None)
        if sparsity is not None and sparsity < 1.0:
            rng = np.random.default_rng(self.config.get('seed', 0))
            n_keep = max(1, int(len(src) * sparsity))
            mask = rng.choice(len(src), size=n_keep, replace=False)
            src = src[mask]
            tgt = tgt[mask]

        return src, tgt
