"""Reproducible RNG for NumPy, PyTorch, and DataLoader workers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU/CUDA if available).

    deterministic_torch: if True, may reduce performance but improve repeatability
    (CUDA deterministic algorithms).
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed % (2**31)))
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def worker_init_fn(worker_id: int, base_seed: Optional[int]) -> None:
    """Initializer for DataLoader workers (reproducible per-worker seed)."""
    if base_seed is None:
        return
    ss_np = np.random.SeedSequence([base_seed, worker_id, 1])
    np.random.seed(ss_np.generate_state(4))
    try:
        import torch

        ss_pt = np.random.SeedSequence([base_seed, worker_id, 2])
        torch.manual_seed(int(ss_pt.generate_state(1)[0]) % (2**31))
    except ImportError:
        pass
