#!/usr/bin/env python
"""
Training entry point.

Usage:
    python scripts/train.py --config config/experiments/exp_stdp_baseline.yaml \\
                            --dataset datasets/vox1_small

    # Resume from checkpoint
    python scripts/train.py --config config/experiments/exp_stdp_baseline.yaml \\
                            --dataset datasets/vox1_small \\
                            --resume-from ckpt_sample_000030
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_config
from snn_core.network import SNNNetwork
from snn_core.training import Trainer
from snn_core.dataset import VoxDataset
from storage.checkpoint_manager import CheckpointManager
from storage.weight_history_store import WeightHistoryStore


def main():
    parser = argparse.ArgumentParser(description='Train SNN with STDP')
    parser.add_argument('--config',          required=True, help='Path to experiment YAML config')
    parser.add_argument('--dataset',         required=True, help='Root directory of audio dataset')
    parser.add_argument('--checkpoint-dir',  default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume-from',     default=None, help='Checkpoint ID to resume from')
    args = parser.parse_args()

    # Load experiment config (merges with base defaults)
    config = load_config(args.config)
    print(f"\nExperiment : {config.name}")
    print(f"Description: {config.description}\n")

    # Build network architecture from config
    network = SNNNetwork(config.network)
    for info in network.layer_info():
        print(f"  Layer   '{info['name']}': {info['N']} neurons ({info['type']})")
    for info in network.synapse_info():
        print(f"  Synapse '{info['name']}': {info['src']} → {info['tgt']} ({info['connectivity']})")
    print()

    # Load dataset
    dataset = VoxDataset(args.dataset, max_samples=config.training.max_samples)
    print(f"Dataset: {dataset}\n")

    # Init trainer
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    weight_history_store = None
    if config.training.save_weights_history:
        wh_path = os.path.join(args.checkpoint_dir, 'weight_history.h5')
        weight_history_store = WeightHistoryStore(wh_path)
        print(f"Weight history: saving to {wh_path}")

    trainer = Trainer(network, config, checkpoint_mgr, weight_history_store)

    # Run
    history = trainer.run_training(dataset, resume_from=args.resume_from)

    # Summary
    for key, values in history.items():
        if values:
            print(f"  {key}: final={values[-1]:.5f}  mean={sum(values)/len(values):.5f}")


if __name__ == '__main__':
    main()
