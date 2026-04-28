"""Shared type definitions for training visualisation."""

import numpy as np

class TrainingSnapshot:
    """Container for live visualisation data.

    Attributes:
        epoch: Current epoch index.
        step: Current batch step index.
        total_epochs: Total epochs in the training run.
        total_steps: Total batch steps per epoch.
        cost: Current cost value.
        accuracy: Current accuracy value.
        grid: Prediction grid reshaped to mesh dimensions.
    """
    def __init__(
        self,
        epoch: int,
        step: int,
        total_epochs: int,
        total_steps: int,
        cost: float,
        accuracy: float,
        grid: np.ndarray,
    ) -> None:
        self.epoch: int = epoch
        self.step: int = step
        self.total_epochs: int = total_epochs
        self.total_steps: int = total_steps
        self.cost: float = cost
        self.accuracy: float = accuracy
        self.grid: np.ndarray = grid
