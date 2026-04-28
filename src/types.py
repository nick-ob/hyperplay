"""Shared type definitions for training visualisation."""

from __future__ import annotations

import numpy as np

class TrainingSnapshot:
    """Container for live visualisation data.

    Attributes:
        epoch: Current epoch index.
        step: Current batch step index.
        grid: Prediction grid reshaped to mesh dimensions.
    """
    def __init__(self, epoch: int, step: int, grid: np.ndarray) -> None:
        self.epoch: int = epoch
        self.step: int = step
        self.grid: np.ndarray = grid
