from pathlib import Path

import numpy as np
import torch


class RunningMoments:
    """Numerically stable running mean/variance for activation vectors."""

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return

        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dim:
            raise ValueError(f"Expected batch shape (n, {self.dim}), got {batch.shape}")

        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + (delta**2) * self.count * batch_count / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("No samples were observed; cannot compute statistics")

        var = self.m2 / self.count
        var = np.maximum(var, 1e-8)
        return self.mean.astype(np.float32), var.astype(np.float32)


def save_rep_statistics(stats: RunningMoments, output_path: Path) -> None:
    mean, var = stats.finalize()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": torch.from_numpy(mean),
            "var": torch.from_numpy(var),
        },
        output_path,
    )
