import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from glp import flow_matching
from glp.denoiser import load_glp
from glp.utils_acts import MemmapReader

from .loading import resolve_device


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices have different shapes")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def rep_fd(feats1: np.ndarray, feats2: np.ndarray) -> float:
    mu1 = np.mean(feats1, axis=0)
    sig1 = np.cov(feats1, rowvar=False)
    mu2 = np.mean(feats2, axis=0)
    sig2 = np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


def load_reference_vectors(data_dir: str, *, sample_size: int, seed: int) -> np.ndarray:
    data_path = Path(data_dir)
    dtype = np.dtype((data_path / "dtype.txt").read_text().strip().replace("np.", ""))
    reader = MemmapReader(data_path, dtype)

    n = min(sample_size, len(reader))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(reader), size=n, replace=False)
    vectors = np.stack([reader[int(i)] for i in selected], axis=0)

    if vectors.dtype == np.int16:
        vectors = torch.from_numpy(vectors).view(torch.bfloat16).float().numpy()
    else:
        vectors = vectors.astype(np.float32)

    return vectors


def evaluate_checkpoint(
    *,
    weights_folder: str,
    reference_data_dir: str,
    checkpoint: str = "final",
    sample_size: int = 50000,
    num_timesteps: int = 1000,
    batch_size: int = 256,
    device: str = "auto",
    seed: int = 42,
    layer_idx: int | None = None,
    save_path: str | None = None,
) -> dict:
    device = resolve_device(device)
    ref_vectors = load_reference_vectors(reference_data_dir, sample_size=sample_size, seed=seed)

    model = load_glp(weights_folder, device=device, checkpoint=checkpoint)
    hidden_size = ref_vectors.shape[1]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(
        (ref_vectors.shape[0], 1, hidden_size),
        generator=generator,
        device=device,
    )

    generated = []
    for start in range(0, noise.shape[0], batch_size):
        end = start + batch_size
        batch = flow_matching.sample(
            model,
            noise[start:end],
            num_timesteps=num_timesteps,
            layer_idx=layer_idx,
        )
        batch = model.normalizer.denormalize(batch, layer_idx=layer_idx)
        generated.append(batch[:, 0, :].float().cpu().numpy())

    gen_vectors = np.concatenate(generated, axis=0)
    fd_value = rep_fd(gen_vectors, ref_vectors)

    metrics = {
        "weights_folder": weights_folder,
        "checkpoint": checkpoint,
        "num_samples": int(ref_vectors.shape[0]),
        "num_timesteps": int(num_timesteps),
        "fd": float(fd_value),
    }

    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with save_file.open("w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
