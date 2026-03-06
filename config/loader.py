"""
Config loader — YAML → validated ExperimentConfig dataclass.

Merges base defaults with experiment-specific overrides using deep merge,
so experiments only need to specify what differs from the base.
"""

import copy
import yaml
import os
from dataclasses import asdict
from .schema import (
    ExperimentConfig, AudioConfig, NetworkConfig, TrainingConfig,
    LayerConfig, SynapseConfig, ConnectivityConfig, LearningRuleConfig,
)

_BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'defaults', 'base.yaml')


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_config(experiment_path: str) -> ExperimentConfig:
    """
    Load an experiment YAML and merge with base defaults.

    Args:
        experiment_path: Path to experiment YAML file.

    Returns:
        Fully populated ExperimentConfig.
    """
    base_raw  = _load_yaml(_BASE_CONFIG_PATH)
    exp_raw   = _load_yaml(experiment_path)
    raw       = _deep_merge(base_raw, exp_raw)

    audio_cfg = AudioConfig(**raw['audio'])

    layers = [
        LayerConfig(
            name=l['name'],
            type=l['type'],
            N_neurons=l['N_neurons'],
            neuron_params=l.get('neuron_params', {}),
        )
        for l in raw['network']['layers']
    ]

    synapses = []
    for s in raw['network']['synapses']:
        conn_raw = s['connectivity']
        lr_raw   = s['learning_rule']

        conn = ConnectivityConfig(
            type=conn_raw['type'],
            sparsity=conn_raw.get('sparsity'),
            graph_source=conn_raw.get('graph_source'),
            seed=conn_raw.get('seed'),
        )

        lr = LearningRuleConfig(
            type=lr_raw['type'],
            taupre=lr_raw.get('taupre'),
            taupost=lr_raw.get('taupost'),
            Apre_delta=lr_raw.get('Apre_delta'),
            Apost_delta=lr_raw.get('Apost_delta'),
            wmax=lr_raw.get('wmax'),
            wmin=lr_raw.get('wmin'),
            w_init_mean=lr_raw.get('w_init_mean'),
            homeostasis_norm=lr_raw.get('homeostasis_norm'),
            seed=lr_raw.get('seed'),
            extra={k: v for k, v in lr_raw.items()
                   if k not in {'type','taupre','taupost','Apre_delta','Apost_delta',
                                'wmax','wmin','w_init_mean','homeostasis_norm','seed'}},
        )

        synapses.append(SynapseConfig(
            name=s['name'],
            src_layer=s['src_layer'],
            tgt_layer=s['tgt_layer'],
            connectivity=conn,
            learning_rule=lr,
        ))

    network_cfg  = NetworkConfig(layers=layers, synapses=synapses)
    training_cfg = TrainingConfig(**raw['training'])

    return ExperimentConfig(
        name=raw.get('name', 'unnamed'),
        description=raw.get('description', ''),
        version=str(raw.get('version', '1.0')),
        audio=audio_cfg,
        network=network_cfg,
        training=training_cfg,
    )


def save_config(config: ExperimentConfig, path: str):
    """Serialise ExperimentConfig back to YAML for reproducibility."""
    data = asdict(config)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
