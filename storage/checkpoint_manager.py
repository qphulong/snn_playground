"""
CheckpointManager — save and load full training state.

Checkpoint format (pickle):
    {
        'weights':    {syn_name: (N_src, N_tgt) float32 array},
        'sample_idx': int,
        'history':    {metric_name: [float]},
        'config':     ExperimentConfig,
        'timestamp':  float (unix time),
    }
"""

import os
import glob
import time
import pickle
from typing import Dict, Any, List, Optional


class CheckpointManager:

    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def save(self, data: Dict[str, Any]) -> str:
        """
        Save checkpoint. Returns checkpoint ID.

        data must contain at minimum:
            'weights', 'sample_idx', 'history', 'config'
        """
        data.setdefault('timestamp', time.time())
        ckpt_id = f"ckpt_sample_{data['sample_idx']:06d}"
        path    = os.path.join(self.dir, f"{ckpt_id}.pkl")

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return ckpt_id

    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint by ID."""
        path = os.path.join(self.dir, f"{checkpoint_id}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint, or None if none exist."""
        ids = self.list_checkpoints()
        if not ids:
            return None
        return self.load(ids[-1])

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of checkpoint IDs (oldest first)."""
        paths = glob.glob(os.path.join(self.dir, 'ckpt_sample_*.pkl'))
        ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        return sorted(ids)

    def get_metadata(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load lightweight metadata without full weight matrices."""
        ckpt = self.load(checkpoint_id)
        return {
            'id':           checkpoint_id,
            'sample_idx':   ckpt.get('sample_idx'),
            'timestamp':    ckpt.get('timestamp'),
            'config_name':  ckpt.get('config', {}).name if hasattr(ckpt.get('config', {}), 'name') else None,
            'history_keys': list(ckpt.get('history', {}).keys()),
        }

    def list_with_metadata(self) -> List[Dict[str, Any]]:
        return [self.get_metadata(cid) for cid in self.list_checkpoints()]
