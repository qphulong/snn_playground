"""
Config schema — dataclasses for all hyperparameters.

These are populated from YAML files by config/loader.py.
No defaults live in code — all defaults are in config/defaults/base.yaml.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AudioConfig:
    sr: int
    num_filters: int
    f_min: float
    sustained_per_band: int
    onset_per_band: int
    phase_per_band: int
    SCALE: float
    percentile: int

    @property
    def neurons_per_band(self) -> int:
        return self.sustained_per_band + self.onset_per_band + self.phase_per_band

    @property
    def N_in(self) -> int:
        return self.num_filters * self.neurons_per_band


@dataclass
class ConnectivityConfig:
    type: str
    # Optional fields — populated only when type requires them
    sparsity: Optional[float] = None
    graph_source: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        valid = {'dense', 'sparse', 'graph'}
        if self.type not in valid:
            # Allow custom types registered in layers/__init__.py
            pass


@dataclass
class LearningRuleConfig:
    type: str
    # STDP params
    taupre:      Optional[float] = None
    taupost:     Optional[float] = None
    Apre_delta:  Optional[float] = None
    Apost_delta: Optional[float] = None
    wmax:        Optional[float] = None
    wmin:        Optional[float] = None
    w_init_mean: Optional[float] = None
    homeostasis_norm: Optional[float] = None
    seed:        Optional[int]   = None
    # Any extra params for custom rules
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerConfig:
    name: str
    type: str
    N_neurons: int
    neuron_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapseConfig:
    name: str
    src_layer: str
    tgt_layer: str
    connectivity: ConnectivityConfig
    learning_rule: LearningRuleConfig


@dataclass
class NetworkConfig:
    layers: List[LayerConfig]
    synapses: List[SynapseConfig]


@dataclass
class TrainingConfig:
    max_samples: int
    checkpoint_interval: int
    dt: float = 0.1       # ms
    seed: int = 42
    save_weights_history: bool = False  # store full weight matrix after every sample


@dataclass
class ExperimentConfig:
    name: str
    description: str
    version: str
    audio: AudioConfig
    network: NetworkConfig
    training: TrainingConfig
