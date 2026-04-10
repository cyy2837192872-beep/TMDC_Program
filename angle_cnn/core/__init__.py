"""Shared utilities for moiré CNN pipeline (fonts, I/O, simulation, degradations, AFM import)."""

from .fonts import setup_matplotlib_cjk_font
from .io_utils import require_file, load_npz_dataset, load_model_checkpoint
from .seed import set_global_seed

__all__ = [
    "setup_matplotlib_cjk_font",
    "require_file",
    "load_npz_dataset",
    "load_model_checkpoint",
    "set_global_seed",
]

# io_afm is imported on-demand (requires optional igor2 dependency)
