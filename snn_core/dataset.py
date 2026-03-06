import glob
import os
from abc import ABC, abstractmethod
from typing import Optional


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        """Return audio file path for the given index."""
        ...


class VoxDataset(Dataset):
    """
    VoxCeleb-style audio dataset.

    Recursively scans root_dir for .wav files, sorted for reproducibility.
    """

    def __init__(self, root_dir: str, max_samples: Optional[int] = None):
        files = sorted(glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True))
        if not files:
            raise ValueError(f"No .wav files found in: {root_dir}")
        self.files = files[:max_samples] if max_samples else files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        return self.files[idx]

    def __repr__(self) -> str:
        return f"VoxDataset({len(self.files)} files)"
