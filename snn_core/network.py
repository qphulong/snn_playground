"""
SNNNetwork — architecture-agnostic Brian2 network.

Structure is fully determined by the ExperimentConfig passed at construction.
No layer sizes, connectivity patterns, or learning rules are hardcoded here.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, TimedArray,
    start_scope, defaultclock, run, ms, Network
)

from .layers import LAYER_TYPES, CONNECTIVITY_TYPES
from .learning_rules import LEARNING_RULE_TYPES


class SNNNetwork:
    """
    Builds and runs a Brian2 SNN from config.

    Layers, connectivity, and learning rules are all resolved from
    LAYER_TYPES / CONNECTIVITY_TYPES / LEARNING_RULE_TYPES registries,
    so adding new types requires no changes here.
    """

    def __init__(self, config: Any):
        """
        Args:
            config: NetworkConfig (see config/schema.py)
        """
        self.config = config
        self.layers = {}       # name → Layer instance
        self.synapses = {}     # name → dict with connectivity + learning_rule
        self._build_from_config()

    # ------------------------------------------------------------------
    # Build phase (runs once, creates reusable layer/synapse instances)
    # ------------------------------------------------------------------

    def _build_from_config(self):
        for layer_cfg in self.config.layers:
            cls = LAYER_TYPES.get(layer_cfg.type)
            if cls is None:
                raise ValueError(
                    f"Unknown layer type '{layer_cfg.type}'. "
                    f"Available: {list(LAYER_TYPES.keys())}"
                )
            self.layers[layer_cfg.name] = cls(
                name=layer_cfg.name,
                N_neurons=layer_cfg.N_neurons,
                config=layer_cfg.neuron_params,
            )

        for syn_cfg in self.config.synapses:
            conn_cls = CONNECTIVITY_TYPES.get(syn_cfg.connectivity.type)
            if conn_cls is None:
                raise ValueError(
                    f"Unknown connectivity type '{syn_cfg.connectivity.type}'. "
                    f"Available: {list(CONNECTIVITY_TYPES.keys())}"
                )
            lr_cls = LEARNING_RULE_TYPES.get(syn_cfg.learning_rule.type)
            if lr_cls is None:
                raise ValueError(
                    f"Unknown learning rule '{syn_cfg.learning_rule.type}'. "
                    f"Available: {list(LEARNING_RULE_TYPES.keys())}"
                )
            self.synapses[syn_cfg.name] = {
                'src':           syn_cfg.src_layer,
                'tgt':           syn_cfg.tgt_layer,
                'connectivity':  conn_cls(vars(syn_cfg.connectivity)),
                'learning_rule': lr_cls(vars(syn_cfg.learning_rule)),
                'config':        syn_cfg,
            }

    # ------------------------------------------------------------------
    # Run phase (Brian2 objects are ephemeral — recreated each call)
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        input_currents: np.ndarray,
        weights: Dict[str, np.ndarray],
        dt_ms: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run one simulation with the given input currents and initial weights.

        Brian2 objects (NeuronGroup, Synapses) are created fresh here and
        discarded after — weight matrices are the only persistent state.

        Args:
            input_currents: (N_in, T) float array from audio_encoding
            weights:        {synapse_name: (N_src, N_tgt) weight matrix}
                            Keys must match config synapse names.
                            Pass empty dict to use learning rule defaults.
            dt_ms:          simulation time step in milliseconds

        Returns:
            {
                'weights': {syn_name: (N_src, N_tgt) updated weight matrix},
                'spikes':  {layer_name: {'times_ms': np.ndarray, 'indices': np.ndarray}},
            }
        """
        start_scope()
        DT = dt_ms * ms
        defaultclock.dt = DT

        _, T = input_currents.shape
        duration = T * DT

        # Create TimedArray for input current injection.
        # Named I_input; referenced in AdaptiveLIF equations as I_input(t, i).
        I_input = TimedArray(input_currents.T.astype(float), dt=DT)

        # ── Build NeuronGroups ──────────────────────────────────────────
        brian2_groups: Dict[str, NeuronGroup] = {}
        for name, layer in self.layers.items():
            ns = layer.get_namespace(dt_ms)
            # Inject I_input for input layers (AdaptiveLIF references it by name)
            if 'I_input' in layer.get_equations():
                ns['I_input'] = I_input

            g = NeuronGroup(
                layer.N,
                model=layer.get_equations(),
                threshold=layer.get_threshold(),
                reset=layer.get_reset(),
                refractory=layer.get_refractory() * ms,
                method='euler',
                namespace=ns,
            )
            # Set initial state
            for var, val in layer.get_initial_state().items():
                setattr(g, var, val)

            brian2_groups[name] = g

        # ── Build Synapses ─────────────────────────────────────────────
        brian2_synapses: Dict[str, Synapses] = {}
        conn_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for syn_name, syn_info in self.synapses.items():
            src_layer = self.layers[syn_info['src']]
            tgt_layer = self.layers[syn_info['tgt']]
            lr        = syn_info['learning_rule']
            conn      = syn_info['connectivity']

            src_group = brian2_groups[syn_info['src']]
            tgt_group = brian2_groups[syn_info['tgt']]

            src_idx, tgt_idx = conn.get_connections(src_layer.N, tgt_layer.N)
            conn_indices[syn_name] = (src_idx, tgt_idx)

            S = Synapses(
                src_group,
                tgt_group,
                model=lr.get_synapse_model(),
                on_pre=lr.get_on_pre(),
                on_post=lr.get_on_post(),
                namespace=lr.get_namespace(),
            )
            S.connect(i=src_idx, j=tgt_idx)

            # Initialise weights
            if syn_name in weights:
                # Load from stored matrix
                w_mat = weights[syn_name]
                S.w = w_mat[src_idx, tgt_idx]
            else:
                # Use learning rule default initialisation
                S.w = lr.get_weight_init(src_layer.N, tgt_layer.N, src_idx, tgt_idx)

            brian2_synapses[syn_name] = S

        # ── Spike monitors ─────────────────────────────────────────────
        # Input layer: count only (record=False) — too many spikes to store per-sample.
        # All other layers: record full spike times and indices for analysis.
        input_layer_name = list(self.layers.keys())[0]
        spike_monitors: Dict[str, SpikeMonitor] = {}
        for name, g in brian2_groups.items():
            record = (name != input_layer_name)
            spike_monitors[name] = SpikeMonitor(g, record=record)

        # ── Run ────────────────────────────────────────────────────────
        net_objects = (
            list(brian2_groups.values())
            + list(brian2_synapses.values())
            + list(spike_monitors.values())
        )
        net = Network(*net_objects)
        net.run(duration)

        # ── Extract results ────────────────────────────────────────────
        updated_weights: Dict[str, np.ndarray] = {}
        for syn_name, S in brian2_synapses.items():
            src_layer = self.layers[self.synapses[syn_name]['src']]
            tgt_layer = self.layers[self.synapses[syn_name]['tgt']]
            src_idx, tgt_idx = conn_indices[syn_name]
            w_mat = np.zeros((src_layer.N, tgt_layer.N), dtype=np.float32)
            w_mat[src_idx, tgt_idx] = np.array(S.w, dtype=np.float32)
            updated_weights[syn_name] = w_mat

        spikes: Dict[str, Dict] = {}
        for name, mon in spike_monitors.items():
            if name == input_layer_name:
                # record=False: only num_spikes is available
                spikes[name] = {
                    'times_ms':   np.array([], dtype=np.float32),
                    'indices':    np.array([], dtype=np.int32),
                    'num_spikes': int(mon.num_spikes),
                }
            else:
                spikes[name] = {
                    'times_ms': np.array(mon.t / ms, dtype=np.float32),
                    'indices':  np.array(mon.i, dtype=np.int32),
                    'num_spikes': int(mon.num_spikes),
                }

        return {
            'weights': updated_weights,
            'spikes':  spikes,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def init_weights(self) -> Dict[str, np.ndarray]:
        """Initialise all synapse weight matrices using learning rule defaults."""
        weights = {}
        for syn_name, syn_info in self.synapses.items():
            src_layer = self.layers[syn_info['src']]
            tgt_layer = self.layers[syn_info['tgt']]
            lr        = syn_info['learning_rule']
            conn      = syn_info['connectivity']
            src_idx, tgt_idx = conn.get_connections(src_layer.N, tgt_layer.N)
            w_vec = lr.get_weight_init(src_layer.N, tgt_layer.N, src_idx, tgt_idx)
            w_mat = np.zeros((src_layer.N, tgt_layer.N), dtype=np.float32)
            w_mat[src_idx, tgt_idx] = w_vec
            weights[syn_name] = w_mat
        return weights

    def layer_info(self) -> list:
        return [
            {'name': l.name, 'N': l.N, 'type': type(l).__name__}
            for l in self.layers.values()
        ]

    def synapse_info(self) -> list:
        result = []
        for name, s in self.synapses.items():
            result.append({
                'name': name,
                'src':  s['src'],
                'tgt':  s['tgt'],
                'connectivity': type(s['connectivity']).__name__,
                'learning_rule': type(s['learning_rule']).__name__,
            })
        return result
