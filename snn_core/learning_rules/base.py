from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class LearningRule(ABC):
    """
    Abstract learning rule.

    Each subclass provides Brian2 synapse code and an optional
    homeostatic normalization applied after each simulation.
    """

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def get_synapse_model(self) -> str:
        """Return Brian2 synapse variable definitions (w, traces, etc.)."""

    @abstractmethod
    def get_on_pre(self) -> str:
        """Return Brian2 on_pre code (executed when pre-synaptic neuron fires)."""

    @abstractmethod
    def get_on_post(self) -> str:
        """Return Brian2 on_post code (executed when post-synaptic neuron fires)."""

    def get_namespace(self) -> dict:
        """Return extra Brian2 namespace variables for the synapse model."""
        return {}

    def get_weight_init(self, N_src: int, N_tgt: int, src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
        """
        Return initial weight vector aligned with (src_idx, tgt_idx).
        Default: uniform random in [wmin, 2 * w_init_mean].
        """
        c = self.config
        wmin = c.get('wmin', 0.0)
        wmax_init = 2 * c.get('w_init_mean', 0.5)
        rng = np.random.default_rng(c.get('seed', 42))
        return rng.uniform(wmin, wmax_init, size=len(src_idx)).astype(np.float32)

    def apply_homeostasis(self, weights: np.ndarray) -> np.ndarray:
        """
        Optional: apply homeostatic normalization to full weight matrix.

        Args:
            weights: (N_src, N_tgt) weight matrix

        Returns:
            Normalized weight matrix of same shape.
        """
        return weights
