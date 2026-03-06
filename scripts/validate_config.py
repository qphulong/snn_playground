#!/usr/bin/env python
"""
Validate an experiment config file without running training.

Usage:
    python scripts/validate_config.py config/experiments/exp_stdp_baseline.yaml
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config
from snn_core.network import SNNNetwork


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_config.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Validating: {config_path}")

    config = load_config(config_path)
    print(f"  name:        {config.name}")
    print(f"  version:     {config.version}")
    print(f"  N_in:        {config.audio.N_in}")
    print(f"  max_samples: {config.training.max_samples}")

    network = SNNNetwork(config.network)
    print("\nLayers:")
    for info in network.layer_info():
        print(f"  {info['name']}: {info['N']} neurons ({info['type']})")

    print("\nSynapses:")
    for info in network.synapse_info():
        src_N = network.layers[info['src']].N
        tgt_N = network.layers[info['tgt']].N
        print(f"  {info['name']}: {info['src']}({src_N}) → {info['tgt']}({tgt_N}) | {info['connectivity']} | {info['learning_rule']}")

    print("\nConfig is valid.")


if __name__ == '__main__':
    main()
