import einops
from einops import repeat
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from itertools import chain
import math
from omegaconf import OmegaConf
import os
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
from types import SimpleNamespace

try:
    from generative_latent_prior.glp import flow_matching
except ImportError:
    from glp import flow_matching


def _canonicalize_normalization_method(method):
    if method is None:
        return "gaussian"
    method = str(method).strip().lower().replace("-", "_")
    if method in {"lognorm"}:
        method = "log_norm"

    aliases = {
        "rms": "rmsnorm",
        "rms_norm": "rmsnorm",
        "zscore": "gaussian",
        "z_score": "gaussian",
    }
    method = aliases.get(method, method)

    if method in {"gaussian", "log_norm", "rmsnorm", "iqr"}:
        return method

    if method == "quantile":
        return "quantile_99"

    if method.startswith("quantile_"):
        quantile_raw = method.split("_", 1)[1]
        quantile_percent = _parse_quantile_percent(quantile_raw)
        return f"quantile_{_format_quantile_percent(quantile_percent)}"

    # Allow shorthand numeric input like "99" or "0.99" for quantile norm.
    try:
        quantile_percent = _parse_quantile_percent(method)
        return f"quantile_{_format_quantile_percent(quantile_percent)}"
    except ValueError:
        pass

    raise ValueError(
        f"Unsupported normalization_method '{method}'. "
        "Expected one of ['gaussian', 'log_norm', 'rmsnorm', 'iqr', 'quantile_XX', '99', '0.99']."
    )


def _parse_quantile_percent(raw_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid quantile specification '{raw_value}'. "
            "Use values like 99, 97, or 0.99."
        ) from exc

    if value <= 1.0:
        value *= 100.0

    if not (0.0 < value < 100.0):
        raise ValueError(
            f"Quantile percent must be in (0, 100), got {value}."
        )
    return value


def _format_quantile_percent(value):
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _normalization_requires_stats(method):
    return _canonicalize_normalization_method(method) != "log_norm"


def _canonicalize_noise_sampling_method(method):
    if method is None:
        return "uniform"

    method = str(method).strip().lower().replace("-", "_")
    aliases = {
        "default": "uniform",
        "rand": "uniform",
        "random": "uniform",
        "optimal_transport": "sot",
        "ot": "sot",
        "hungarian": "sot",
        "sinkhorn": "sinkhorn",
        "sinkhorn_ot": "sinkhorn",
    }
    method = aliases.get(method, method)

    if method in {"uniform", "sot", "sinkhorn"}:
        return method

    raise ValueError(
        f"Unsupported noise_sampling_method '{method}'. "
        "Expected one of ['uniform', 'sot', 'sinkhorn']."
    )



def _canonicalize_u_sampling_method(method):
    if method is None:
        return "uniform"

    method = str(method).strip().lower().replace("-", "_")
    aliases = {
        "default": "uniform",
        "rand": "uniform",
        "random": "uniform",
    }
    method = aliases.get(method, method)

    if method in {"uniform", "beta", "logit_normal"}:
        return method

    raise ValueError(
        f"Unsupported u_sampling_method '{method}'. "
        "Expected one of ['uniform', 'beta', 'logit_normal']."
    )


def _canonicalize_ot_chunk_size(chunk_size):
    if chunk_size is None:
        return 4096

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"ot_chunk_size must be > 0, got {chunk_size}.")
    return chunk_size


_SINKHORN_COLLISION_WARNING_EMITTED = False


def greedy_assignment_gpu(cost_matrix):
    """Fallback O(N^2) greedy assignment if Sinkhorn collisions occur."""
    n_items = cost_matrix.shape[0]
    matched_j = torch.zeros(n_items, dtype=torch.long, device=cost_matrix.device)
    available_j = torch.ones(n_items, dtype=torch.bool, device=cost_matrix.device)

    for row_idx in range(n_items):
        valid_costs = cost_matrix[row_idx].clone()
        valid_costs[~available_j] = float("inf")
        best_j = torch.argmin(valid_costs)
        matched_j[row_idx] = best_j
        available_j[best_j] = False

    return matched_j


def _repair_duplicate_sinkhorn_matches(log_coupling, cost_matrix, best_matches_idx):
    n_items = best_matches_idx.shape[0]
    counts = torch.bincount(best_matches_idx, minlength=n_items)
    if torch.all(counts <= 1):
        return best_matches_idx

    global _SINKHORN_COLLISION_WARNING_EMITTED
    unresolved_count = int((counts[best_matches_idx] > 1).sum().item() - (counts > 1).sum().item())
    if not _SINKHORN_COLLISION_WARNING_EMITTED:
        print(
            "WARNING: Sinkhorn algorithm produced non-unique matches; "
            f"repairing {unresolved_count} collided rows with greedy assignment."
        )
        _SINKHORN_COLLISION_WARNING_EMITTED = True

    repaired_idx = torch.full_like(best_matches_idx, -1)
    used_cols = torch.zeros(n_items, dtype=torch.bool, device=best_matches_idx.device)

    for col_idx in torch.nonzero(counts > 0, as_tuple=False).flatten():
        rows = torch.nonzero(best_matches_idx == col_idx, as_tuple=False).flatten()
        if rows.numel() == 1:
            winner_row = rows[0]
        else:
            winner_row = rows[torch.argmax(log_coupling[rows, col_idx])]
        repaired_idx[winner_row] = col_idx
        used_cols[col_idx] = True

    remaining_rows = torch.nonzero(repaired_idx < 0, as_tuple=False).flatten()
    if remaining_rows.numel() > 0:
        remaining_cols = torch.nonzero(~used_cols, as_tuple=False).flatten()
        local_costs = cost_matrix.index_select(0, remaining_rows).index_select(1, remaining_cols)
        local_matches = greedy_assignment_gpu(local_costs)
        repaired_idx[remaining_rows] = remaining_cols[local_matches]

    return repaired_idx


def minibatch_sinkhorn_ot(x0, x1, epsilon=0.05, iterations=100):
    """Entropic OT pairing between noise x0 and data x1 on GPU."""
    if x0.shape != x1.shape:
        raise ValueError(f"OT pairing requires equal shapes, got {x0.shape} vs {x1.shape}.")

    n_items, _ = x0.shape
    if n_items < 2:
        return x1

    epsilon = float(epsilon)
    iterations = int(iterations)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}.")

    cost_matrix = torch.cdist(x0.float(), x1.float(), p=2).pow(2)
    cost_scale = cost_matrix.detach().mean().clamp_min(torch.finfo(cost_matrix.dtype).eps)
    cost_matrix = cost_matrix / cost_scale
    log_mass = torch.log(torch.tensor(1.0 / n_items, device=cost_matrix.device, dtype=cost_matrix.dtype))
    u = torch.zeros(n_items, device=cost_matrix.device, dtype=cost_matrix.dtype)
    v = torch.zeros(n_items, device=cost_matrix.device, dtype=cost_matrix.dtype)

    for _ in range(iterations):
        u = epsilon * (
            log_mass - torch.logsumexp((v.unsqueeze(0) - cost_matrix) / epsilon, dim=1)
        )
        v = epsilon * (
            log_mass - torch.logsumexp((u.unsqueeze(1) - cost_matrix) / epsilon, dim=0)
        )

    log_coupling = (u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / epsilon
    best_matches_idx = torch.argmax(log_coupling, dim=1)
    best_matches_idx = _repair_duplicate_sinkhorn_matches(log_coupling, cost_matrix, best_matches_idx)

    return x1[best_matches_idx]


def _match_noise_to_latents_ot_chunk(flat_latents, flat_noise, *, method, chunk_size=4096, epsilon=0.05, iterations=100):
    chunk_size = _canonicalize_ot_chunk_size(chunk_size)
    reordered_noise_chunks = []

    for start_idx in range(0, flat_latents.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, flat_latents.shape[0])
        chunk_latents = flat_latents[start_idx:end_idx]
        chunk_noise = flat_noise[start_idx:end_idx]

        if method == "sot":
            reordered_noise_chunks.append(sliced_optimal_transport(chunk_latents, chunk_noise))
        elif method == "sinkhorn":
            reordered_noise_chunks.append(
                minibatch_sinkhorn_ot(
                    chunk_latents,
                    chunk_noise,
                    epsilon=epsilon,
                    iterations=iterations,
                )
            )
        else:
            raise ValueError(f"Unsupported OT method '{method}'. Expected 'sot' or 'sinkhorn'.")

    return torch.cat(reordered_noise_chunks, dim=0)


def sliced_optimal_transport(x0, x1):
    """
    Sliced Optimal Transport pairing.
    O(N log N) time, O(N) memory.

    Args:
        x0: Gaussian noise batch [N, D]
        x1: Target latent batch [N, D]
    Returns:
        x1_paired: The latent batch reordered to match x0
    """
    if x0.shape != x1.shape:
        raise ValueError(f"OT pairing requires equal shapes, got {x0.shape} vs {x1.shape}.")

    n_items, d_model = x0.shape
    if n_items < 2:
        return x1

    direction = torch.randn(d_model, device=x0.device, dtype=x0.dtype)
    direction = direction / torch.clamp(direction.norm(), min=torch.finfo(direction.dtype).eps)

    proj_x0 = torch.matmul(x0, direction)
    proj_x1 = torch.matmul(x1, direction)

    _, indices_x0 = torch.sort(proj_x0)
    _, indices_x1 = torch.sort(proj_x1)

    inverse_indices_x0 = torch.empty_like(indices_x0)
    inverse_indices_x0[indices_x0] = torch.arange(n_items, device=x0.device)

    matched_indices_x1 = indices_x1[inverse_indices_x0]
    return x1[matched_indices_x1]


def match_noise_to_latents_ot(
    latents,
    noise,
    *,
    method="sot",
    chunk_size=4096,
    epsilon=0.05,
    iterations=100,
):
    if latents.shape != noise.shape:
        raise ValueError(
            f"Latents/noise shape mismatch for OT sampling: {latents.shape} vs {noise.shape}."
        )
    if latents.ndim != 3:
        raise ValueError(
            f"OT sampling expects sequence-shaped latents with shape (batch, seq, dim), got {latents.shape}."
        )

    batch_size = latents.shape[0]
    if batch_size < 2:
        return noise

    # Match whole sequences per batch item, not individual tokens. This keeps
    # the OT cost manageable while letting us process large batches in chunks.
    flat_latents = latents.detach().reshape(batch_size, -1)
    flat_noise = noise.reshape(batch_size, -1)
    optimized_flat_noise = _match_noise_to_latents_ot_chunk(
        flat_latents,
        flat_noise,
        method=_canonicalize_noise_sampling_method(method),
        chunk_size=chunk_size,
        epsilon=epsilon,
        iterations=iterations,
    )
    return optimized_flat_noise.reshape_as(noise)


def _match_noise_to_latents_ot(latents, noise, *, chunk_size=4096, epsilon=0.05, iterations=100):
    return match_noise_to_latents_ot(
        latents,
        noise,
        method="sot",
        chunk_size=chunk_size,
        epsilon=epsilon,
        iterations=iterations,
    )


def _sample_training_noise(
    latents,
    *,
    generator=None,
    noise_sampling_method="uniform",
    ot_chunk_size=4096,
):
    noise_sampling_method = _canonicalize_noise_sampling_method(noise_sampling_method)
    
    noise = torch.randn(
        latents.shape,
        device=latents.device,
        dtype=latents.dtype,
        generator=generator,
    )
    
    if noise_sampling_method in {"sot", "sinkhorn"}:
        return match_noise_to_latents_ot(
            latents,
            noise,
            method=noise_sampling_method,
            chunk_size=ot_chunk_size,
        )

    return noise


def _sample_training_u(latents, *, generator=None, u_sampling_method="uniform"):
    u_sampling_method = _canonicalize_u_sampling_method(u_sampling_method)

    if u_sampling_method == "beta":
        # Beta(5, 1) has inverse CDF x = u^(1/5), so we can keep it fully seedable.
        return torch.rand(latents.shape[0], device=latents.device, generator=generator).pow(1.0 / 5.0)

    if u_sampling_method == "logit_normal":
        normal_samples = torch.randn(
            latents.shape[0],
            device=latents.device,
            generator=generator,
        )
        return torch.sigmoid(normal_samples)

    return torch.rand(latents.shape[0], device=latents.device, generator=generator)


def _resolve_variance_split_indices(variances, tail_proportion=0.05):
    tail_proportion = float(tail_proportion)
    if not (0.0 < tail_proportion < 1.0):
        raise ValueError(f"tail_proportion must be in (0, 1), got {tail_proportion}.")

    variances = variances.detach().float().reshape(-1)
    d_input = int(variances.numel())
    if d_input < 2:
        raise ValueError("Variance-based split requires d_input >= 2.")

    tail_count = max(1, int(round(d_input * tail_proportion)))
    tail_count = min(tail_count, d_input - 1)
    tail_indices = torch.topk(variances, k=tail_count, largest=True).indices.sort().values
    sem_mask = torch.ones(d_input, dtype=torch.bool, device=variances.device)
    sem_mask[tail_indices] = False
    sem_indices = torch.arange(d_input, device=variances.device)[sem_mask]
    return tail_indices, sem_indices


def _compute_std_weighted_loss_from_squared_error(squared_error, tail_indices, sem_indices, tail_weights):
    tail_indices = torch.as_tensor(tail_indices, device=squared_error.device, dtype=torch.long).reshape(-1)
    sem_indices = torch.as_tensor(sem_indices, device=squared_error.device, dtype=torch.long).reshape(-1)
    tail_weights = torch.as_tensor(tail_weights, device=squared_error.device, dtype=squared_error.dtype).reshape(-1)

    tail_error = squared_error.index_select(dim=-1, index=tail_indices)
    if tail_weights.numel() == 1:
        tail_weights = tail_weights.expand(tail_indices.numel())
    if tail_weights.numel() != tail_indices.numel():
        raise ValueError(
            f"tail_weights must have 1 or {tail_indices.numel()} elements, got {tail_weights.numel()}."
        )

    tail_weighted_loss = (tail_error * tail_weights.view(1, 1, -1)).mean()
    tail_region_mse = tail_error.mean()
    sem_region_mse = squared_error.index_select(dim=-1, index=sem_indices).mean()
    loss = tail_weighted_loss + sem_region_mse
    return loss, tail_weighted_loss, tail_region_mse, sem_region_mse


def compute_balanced_loss(v_pred, v_target, tail_indices, sem_indices, tail_weights):
    if v_pred.shape != v_target.shape:
        raise ValueError(f"Balanced loss expects equal shapes, got {v_pred.shape} vs {v_target.shape}.")

    squared_error = (v_pred - v_target).pow(2)
    return _compute_std_weighted_loss_from_squared_error(squared_error, tail_indices, sem_indices, tail_weights)


# ==========================
#     Normalizer Class
# ==========================
class Normalizer(nn.Module):
    def __init__(self, mean, var, normalization_method="gaussian"):
        super().__init__()
        self.mean = nn.Buffer(mean)
        self.var = nn.Buffer(var)
        self.normalization_method = _canonicalize_normalization_method(normalization_method)
    
    def get_layer_stat(self, stat, layer_idx=None):
        if stat.ndim > 1 and stat.shape[0] != 1:
            assert layer_idx is not None, "Layer index must be provided for multi-layer normalization"
        if layer_idx is not None and stat.ndim == 2:
            stat = stat[layer_idx]
            if stat.ndim == 1:
                stat = stat[None, None, :]
            elif stat.ndim == 2:
                stat = stat[:, None, :]
            return stat
        else:
            return stat

    def normalize(self, rep, layer_idx=None):
        if self.normalization_method == "log_norm":
            rep = rep.to(self.var.device)
            return torch.sign(rep) * torch.log1p(torch.abs(rep))

        mean = self.get_layer_stat(self.mean, layer_idx)
        var = self.get_layer_stat(self.var, layer_idx)
        var = torch.clamp(var, min=1e-8)
        scale = torch.sqrt(var)

        if (
            self.normalization_method == "gaussian"
            or self.normalization_method == "iqr"
            or self.normalization_method.startswith("quantile_")
        ):
            return (rep.to(mean.device) - mean) / scale
        if self.normalization_method == "rmsnorm":
            return rep.to(scale.device) / scale

        raise ValueError(f"Unsupported normalization_method '{self.normalization_method}'")
    
    def denormalize(self, rep, layer_idx=None):
        if self.normalization_method == "log_norm":
            rep = rep.to(self.var.device)
            return torch.sign(rep) * torch.expm1(torch.abs(rep))

        mean = self.get_layer_stat(self.mean, layer_idx)
        var = self.get_layer_stat(self.var, layer_idx)
        var = torch.clamp(var, min=1e-8)
        scale = torch.sqrt(var)

        if (
            self.normalization_method == "gaussian"
            or self.normalization_method == "iqr"
            or self.normalization_method.startswith("quantile_")
        ):
            return rep.to(var.device) * scale + mean
        if self.normalization_method == "rmsnorm":
            return rep.to(scale.device) * scale

        raise ValueError(f"Unsupported normalization_method '{self.normalization_method}'")
    
    def check_normalized(self, rep, atol=2.0):
        if self.normalization_method != "gaussian":
            if not torch.isfinite(rep).all():
                print("WARNING: Latents contain non-finite values after normalization.")
            return

        # the tolerance is lenient to catch egregious cases
        rep_mean = rep.view(-1, rep.shape[-1]).mean(dim=0)
        rep_var = rep.view(-1, rep.shape[-1]).var(dim=0, unbiased=False)
        ref_mean = torch.zeros(rep.shape[-1], device=rep.device, dtype=rep.dtype)
        ref_var = torch.ones(rep.shape[-1], device=rep.device, dtype=rep.dtype)
        is_normalized = torch.isclose(rep_mean, ref_mean, atol=atol).all() and torch.isclose(rep_var, ref_var, atol=atol).all()
        if not is_normalized:
            print(
                f"WARNING: Latents may not be normalized "
                f"(expected mean=0 and var=1, got mean={rep_mean.mean().item():.4f} and var={rep_var.mean().item():.4f}). "
                f"Small deviations are expected, but variances much larger than 1 are unusual."
            )

    @classmethod
    def from_config(cls, rep_statistic="", d_input=None, normalization_method="gaussian"):
        normalization_method = _canonicalize_normalization_method(normalization_method)
        if rep_statistic:
            rep_statistic_pt = torch.load(rep_statistic, map_location="cpu")
            rep_mean = rep_statistic_pt["mean"]
            rep_var = rep_statistic_pt["var"]
            saved_method = rep_statistic_pt.get("normalization_method")
            if saved_method is not None:
                saved_method = _canonicalize_normalization_method(saved_method)
                if normalization_method == "gaussian" and saved_method != normalization_method:
                    normalization_method = saved_method
            return cls(rep_mean, rep_var, normalization_method=normalization_method)
        
        
        dim = d_input if d_input is not None else 1
        return cls(
            torch.zeros(dim),
            torch.ones(dim),
            normalization_method=normalization_method,
        )

    def save_config(self, path):
        path = Path(path)
        torch.save(
            {
                "mean": self.mean,
                "var": self.var,
                "normalization_method": self.normalization_method,
            },
            path / f"rep_statistics.pt",
        )

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.    
    Reference: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L41
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class TransformerMLPBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_mlp,
        d_input,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_input = d_input

        self.up_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.time_proj = nn.Linear(d_model, d_mlp)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, t_emb):
        resid_x = x
        post_ln_x = self.ln(x)
        # project up
        interm_x = self.up_proj(post_ln_x)
        # start SwiGLU gate
        g = self.gate_proj(post_ln_x)
        # multiplicative timestep conditioning
        t_emb = self.time_proj(t_emb)
        merged = g * t_emb
        # continue SwiGLU gate
        x = self.act(merged) * interm_x
        # project down
        x = self.down_proj(x)
        return x + resid_x


def _normalize_tail_indices(tail_indices, d_input):
    if tail_indices is None:
        return []

    if torch.is_tensor(tail_indices):
        tail_indices = tail_indices.detach().cpu().tolist()

    tail_indices = [int(idx) for idx in tail_indices]
    if len(set(tail_indices)) != len(tail_indices):
        raise ValueError("split_tail_indices contains duplicate indices.")

    invalid_indices = [idx for idx in tail_indices if idx < 0 or idx >= d_input]
    if invalid_indices:
        raise ValueError(
            f"split_tail_indices contains out-of-range indices for d_input={d_input}: "
            f"{invalid_indices[:10]}"
        )

    return sorted(tail_indices)


class SplitOutputProj(nn.Module):
    def __init__(self, d_model, d_input, tail_indices):
        super().__init__()
        tail_indices = _normalize_tail_indices(tail_indices, d_input)
        if not (0 < len(tail_indices) < d_input):
            raise ValueError(
                f"SplitOutputProj requires 0 < len(tail_indices) < d_input, "
                f"got {len(tail_indices)} for d_input={d_input}."
            )

        tail_indices = torch.tensor(tail_indices, dtype=torch.long)
        mask = torch.ones(d_input, dtype=torch.bool)
        mask[tail_indices] = False

        self.d_model = d_model
        self.d_input = d_input
        self.register_buffer("tail_indices", tail_indices, persistent=False)
        self.register_buffer("nontail_indices", torch.arange(d_input)[mask], persistent=False)
        self.tail_proj = nn.Linear(d_model, int(self.tail_indices.numel()))
        self.nontail_proj = nn.Linear(d_model, int(self.nontail_indices.numel()))

    def forward(self, x):
        tail_out = self.tail_proj(x)
        nontail_out = self.nontail_proj(x)
        out = tail_out.new_empty(*x.shape[:-1], self.d_input)
        out[..., self.tail_indices] = tail_out
        out[..., self.nontail_indices] = nontail_out
        return out


class TransformerMLPDenoiser(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_mlp=1536,
        d_input=1536,
        n_layers=12,
        multi_layer_n_layers=None,
        split=False,
        split_tail_indices=None,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_input = d_input
        self.n_layers = n_layers
        self.multi_layer_n_layers = multi_layer_n_layers
        self.split = False
        self.split_tail_indices = []

        self.layers = nn.ModuleList([
            TransformerMLPBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                d_input=d_input,
            ) for _ in range(n_layers)
        ])
        self.in_proj = nn.Linear(d_input, d_model)
        self.out_proj = nn.Linear(d_model, d_input)
        if split:
            self.configure_split_output(split_tail_indices)

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if multi_layer_n_layers is not None:
            self.layer_embed = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.layer_embed = nn.Identity()
        self.ln = nn.LayerNorm(d_model)

    def configure_split_output(self, tail_indices):
        tail_indices = _normalize_tail_indices(tail_indices, self.d_input)
        use_split = 0 < len(tail_indices) < self.d_input

        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else None
        dtype = ref_param.dtype if ref_param is not None else None

        if use_split:
            out_proj = SplitOutputProj(self.d_model, self.d_input, tail_indices)
        else:
            out_proj = nn.Linear(self.d_model, self.d_input)

        if device is not None:
            out_proj = out_proj.to(device=device, dtype=dtype)

        self.out_proj = out_proj
        self.split = use_split
        self.split_tail_indices = tail_indices if use_split else []
        return self.split_tail_indices

    def forward(self, latents, timesteps, layer_idx=None, **kwargs):
        assert latents.ndim == 2, f"Expected (batch, dim), got shape {latents.shape}"
        x = latents
        # prepare sinusoidal timestep embedding
        timesteps = timesteps.flatten().to(x.device)
        assert timesteps.shape == (x.shape[0],)
        # Keep embedding dtype aligned with model activations to avoid mixed-precision linear errors.
        t_emb = timestep_embedding(timesteps, self.d_model, repeat_only=False).to(dtype=x.dtype)
        emb = self.time_embed(t_emb)
        # prepare sinusoidal layer depth embedding
        use_layer_embed = self.multi_layer_n_layers is not None and layer_idx is not None
        if use_layer_embed:
            if self.multi_layer_n_layers <= 1:
                raise ValueError("multi_layer_n_layers must be > 1 when using layer_idx")
            layer_depth = layer_idx.float() / (self.multi_layer_n_layers - 1)
            layer_emb = timestep_embedding(layer_depth, self.d_model, repeat_only=False).to(dtype=x.dtype)
            emb += self.layer_embed(layer_emb)
        # apply MLP blocks
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, emb)
        x = self.ln(x)
        x = self.out_proj(x)
        return x

class Denoiser(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TransformerMLPDenoiser(**kwargs)
        self.device, self.dtype = None, None

    def configure_split_output(self, tail_indices):
        resolved_indices = self.model.configure_split_output(tail_indices)
        if self.device is not None:
            self.model.out_proj.to(device=self.device, dtype=self.dtype)
        return resolved_indices

    def forward(self, latents, layer_idx=None, **kwargs):
        layer_idx = torch.full((latents.shape[0],), layer_idx, device=latents.device) if isinstance(layer_idx, int) else layer_idx
        # move device and dtype
        device, dtype = latents.device, latents.dtype
        latents = latents.to(device=self.device, dtype=self.dtype)
        # reshape to (batch*seq, dim) 
        # since denoiser does single-token modeling
        b, s, d = latents.shape
        latents = einops.rearrange(latents, "b s d -> (b s) d")
        latents = self.model(latents, layer_idx=layer_idx, **kwargs)
        # reshape back to (batch, seq, dim)
        latents = einops.rearrange(latents, "(b s) d -> b s d", b=b, s=s)
        latents = latents.to(device=device, dtype=dtype)
        return latents
    
    def save_pretrained(self, path, name=None):
        path = Path(path)
        name = name or "mlp"
        save_file(self.state_dict(), path / f"{name}.safetensors")
        
    def load_pretrained(self, path, name=None):
        path = Path(path)
        name = name or "mlp"
        self.load_state_dict(load_file(path / f"{name}.safetensors"))

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        param = next(chain(self.model.parameters(), self.model.buffers()), None)
        self.device = param.device if param is not None else None
        self.dtype = param.dtype if param is not None else None
        return result

# ==========================
#    GLP Wrapper Class
# ==========================
class GLP(nn.Module):
    def __init__(
        self,
        normalizer_config,
        denoiser_config,
        tracedict_config=None,
        noise_sampling_method="uniform",
        u_sampling_method="uniform",
        ot_chunk_size=4096,
        solver=None,
    ):
        super().__init__()
        self.normalizer = Normalizer.from_config(**normalizer_config)
        self.denoiser = Denoiser(**denoiser_config)
        self.scheduler = flow_matching.fm_scheduler()
        self.solver = flow_matching.canonicalize_solver(solver) if solver is not None else None
        self.inference_scheduler = None
        self.tracedict_config = tracedict_config
        self.noise_sampling_method = _canonicalize_noise_sampling_method(noise_sampling_method)
        self.u_sampling_method = _canonicalize_u_sampling_method(u_sampling_method)
        self.ot_chunk_size = _canonicalize_ot_chunk_size(ot_chunk_size)

    def set_inference_solver(self, solver):
        self.solver = flow_matching.canonicalize_solver(solver)
        num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps", 1000)
        self.inference_scheduler = flow_matching.build_inference_scheduler(
            self.solver,
            num_train_timesteps=num_train_timesteps,
        )
        return self

    def configure_split_output_from_normalizer(self, proportion=0.1):
        proportion = float(proportion)
        if not (0.0 < proportion < 1.0):
            raise ValueError(f"split_proportion must be in (0, 1), got {proportion}.")
        if not _normalization_requires_stats(self.normalizer.normalization_method):
            raise ValueError(
                f"Automatic split-output detection requires normalization statistics; "
                f"got normalization_method='{self.normalizer.normalization_method}'."
            )

        var = self.normalizer.var.detach().float()
        if var.ndim == 2:
            if var.shape[0] != 1:
                raise ValueError(
                    "Automatic split-output detection only supports single-layer normalization stats."
                )
            var = var[0]
        elif var.ndim != 1:
            var = var.reshape(-1)

        d_input = self.denoiser.model.d_input
        if var.numel() != d_input:
            raise ValueError(
                f"Normalizer variance shape does not match denoiser d_input: {var.numel()} vs {d_input}."
            )
        if d_input < 2:
            raise ValueError("Split output projection requires d_input >= 2.")

        tail_indices, _ = _resolve_variance_split_indices(var, tail_proportion=proportion)
        return self.denoiser.configure_split_output(tail_indices)

    def _get_balanced_loss_indices(self, proportion=0.05):
        proportion = float(proportion)
        if self.denoiser.model.split_tail_indices:
            tail_indices = torch.as_tensor(
                self.denoiser.model.split_tail_indices,
                dtype=torch.long,
                device=self.normalizer.var.device,
            )
            d_input = int(self.denoiser.model.d_input)
            sem_mask = torch.ones(d_input, dtype=torch.bool, device=tail_indices.device)
            sem_mask[tail_indices] = False
            sem_indices = torch.arange(d_input, device=tail_indices.device)[sem_mask]
            return tail_indices, sem_indices

        var = self.normalizer.var.detach().float()
        if var.ndim == 2:
            if var.shape[0] != 1:
                raise ValueError(
                    "Automatic balanced-loss detection only supports single-layer normalization stats."
                )
            var = var[0]
        elif var.ndim != 1:
            var = var.reshape(-1)

        d_input = self.denoiser.model.d_input
        if var.numel() != d_input:
            raise ValueError(
                f"Normalizer variance shape does not match denoiser d_input: {var.numel()} vs {d_input}."
            )

        return _resolve_variance_split_indices(var, tail_proportion=proportion)

    def save_pretrained(self, path, name=None):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        self.denoiser.save_pretrained(path, name=name)
        self.normalizer.save_config(path)

    def load_pretrained(self, path, name=None):
        path = Path(path)
        self.denoiser.load_pretrained(path, name=name)

    def forward(
        self,
        *,
        latents: torch.FloatTensor,                       # (batch, seq, dim)
        u: torch.FloatTensor | float | None = None,       # (batch,) or scalar
        layer_idx: torch.LongTensor | int | None = None,  # (batch,) or scalar
        loss_kwargs: dict | None = None,
        generator: torch.Generator | None = None,
        global_step: int | None = None,
        total_steps: int | None = None,
        two_phase: bool = False,
        **kwargs
    ) -> SimpleNamespace:
        # prepare extra params
        assert latents.ndim == 3, f"Expected (batch, seq, dim), got shape {latents.shape}"
        self.normalizer.check_normalized(latents)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        u = torch.full((latents.shape[0],), u, device=latents.device) if isinstance(u, float) else u

        if u is None:
            u = _sample_training_u(
                latents,
                generator=generator,
                u_sampling_method=self.u_sampling_method,
            )

        # prepare flow matching inputs and target
        noise = _sample_training_noise(
            latents,
            generator=generator,
            noise_sampling_method=self.noise_sampling_method,
            ot_chunk_size=self.ot_chunk_size,
        )
        noisy_latents, target, timesteps, meta = flow_matching.fm_prepare(
            self.scheduler,
            latents,
            noise,
            u=u,
            generator=generator
        )
        # compute denoiser forward pass
        outputs = self.denoiser(
            latents=noisy_latents,
            timesteps=timesteps,
            layer_idx=layer_idx,
            **kwargs
        )
        
        outputs_f32 = outputs.float()
        target_f32 = target.float()

        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        tail_proportion = float(loss_kwargs.get("tail_variance_proportion", 0.05) or 0.05)

        raw_outputs = self.normalizer.denormalize(outputs_f32, layer_idx=layer_idx)
        raw_target = self.normalizer.denormalize(target_f32, layer_idx=layer_idx)
        raw_error = (raw_outputs - raw_target).pow(2)

        tail_indices, sem_indices = self._get_balanced_loss_indices(proportion=tail_proportion)
        norm_error = (outputs_f32 - target_f32).pow(2)
        norm_var = self.normalizer.get_layer_stat(self.normalizer.var, layer_idx).detach().float().reshape(-1)
        tail_weights = norm_var.index_select(0, tail_indices.to(device=norm_var.device)).clamp_min(1e-8).sqrt()
        loss, tail_weighted_mse, tail_region_mse, non_tail_region_mse = _compute_std_weighted_loss_from_squared_error(
            norm_error,
            tail_indices.to(device=norm_error.device),
            sem_indices.to(device=norm_error.device),
            tail_weights.to(device=norm_error.device),
        )
        loss_raw = raw_error.mean().detach()
        tail_dim_fraction = outputs_f32.new_tensor(tail_indices.numel() / float(self.denoiser.model.d_input))
        tail_weight_mean = tail_weights.mean().detach()
        tail_weight_max = tail_weights.max().detach()

        # Legacy tail-aware weighting, kept in comment for future reuse.
        # loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        # mse_element = torch.nn.functional.mse_loss(
        #     outputs_f32,
        #     target_f32,
        #     reduction='none',
        # )
        # loss_unreduced = mse_element.view(latents.shape[0], -1).mean(dim=-1)
        # tail_base_mse = loss_unreduced.detach().mean()
        # tail_aware_weight = float(loss_kwargs.get("tail_aware_weight", 0.0) or 0.0)
        # tail_aware_min_weight = float(loss_kwargs.get("tail_aware_min_weight", 0.1) or 0.0)
        # tail_aware_max_weight = float(loss_kwargs.get("tail_aware_max_weight", 10.0) or 0.0)
        # tail_aware_start = int(loss_kwargs.get("tail_aware_start", 1000) or 0)
        # tail_fraction = outputs_f32.new_tensor(0.0)
        # tail_weight_mean = outputs_f32.new_tensor(1.0)
        # tail_weight_max = outputs_f32.new_tensor(1.0)
        # tail_region_mse = outputs_f32.new_tensor(0.0)
        # non_tail_region_mse = outputs_f32.new_tensor(0.0)
        # with torch.no_grad():
        #     raw_latents = self.normalizer.denormalize(latents, layer_idx=layer_idx).detach().float()
        #     tail_mask = None
        # if tail_aware_weight > 0.0:
        #     with torch.no_grad():
        #         raw_magnitude = raw_latents.abs()
        #         batch_mean_mag = raw_magnitude.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        #         full_multiplier = (raw_magnitude / batch_mean_mag).pow(tail_aware_weight)
        #         tail_weight_mean = full_multiplier.mean()
        #         tail_weight_max = full_multiplier.max()
        #         if tail_aware_min_weight > 0.0 or tail_aware_max_weight > 0.0:
        #             clamp_kwargs = {}
        #             if tail_aware_min_weight > 0.0:
        #                 clamp_kwargs["min"] = tail_aware_min_weight
        #             if tail_aware_max_weight > 0.0:
        #                 clamp_kwargs["max"] = tail_aware_max_weight
        #             full_multiplier = full_multiplier.clamp(**clamp_kwargs)
        #         if global_step is not None and tail_aware_start > 0 and global_step < tail_aware_start:
        #             blend_factor = float(global_step) / float(tail_aware_start)
        #         else:
        #             blend_factor = 1.0
        #         tail_multiplier = 1.0 + (full_multiplier - 1.0) * blend_factor
        #         tail_mask = tail_multiplier > 1.0
        #         tail_fraction = tail_mask.float().mean()
        #     loss_unreduced = (mse_element * tail_multiplier.to(mse_element.dtype)).view(latents.shape[0], -1).mean(dim=-1)
        # tail_weighted_mse = loss_unreduced.detach().mean()
        # loss = loss_unreduced.mean()

        # ===== proper metrics =====
        raw_latents = self.normalizer.denormalize(latents, layer_idx=layer_idx).detach().float().view(-1, latents.shape[-1])
        raw_outputs = raw_outputs.detach().view(-1, outputs.shape[-1])
        raw_target = raw_target.detach().view(-1, target.shape[-1])

        # relative squared error  
        pred = outputs.view(-1, outputs.shape[-1])
        tgt  = target.view(-1, target.shape[-1])
        tgt_norm = tgt.norm(dim=-1, keepdim=True) + 1e-6
        latent_pre_l2 = raw_latents.norm(dim=-1, keepdim=True) + 1e-6
        latent_post_l2 = latents.norm(dim=-1, keepdim=True) + 1e-6
        pre_l2_std = latent_pre_l2.std().item()
        post_l2_std = latent_post_l2.std().item()
        latent_pre_l1 = raw_latents.norm(dim=-1, keepdim=True, p=1) + 1e-6
        latent_post_l1 = latents.norm(dim=-1, keepdim=True, p=1) + 1e-6
        tgt_norm_sq = (tgt ** 2).sum(dim=-1) + 1e-8
        loss_rel = ((pred - tgt) ** 2).sum(dim=-1) / tgt_norm_sq
        loss_rel = loss_rel.mean()

        # raw-space MSE (THIS is comparable across normalization)

        # cosine similarity (KEEP)
        cos_sim = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1).mean()



        with torch.no_grad():
            # --- Timestep Loss Mask ---
            # Track the current normalized MSE by u buckets for debugging and reporting.
            u_flat = meta["u"].view(-1).to(device=pred.device)
            loss_unreduced = norm_error.view(latents.shape[0], -1).mean(dim=-1)

            mask_early = u_flat < 0.3
            mask_mid = (u_flat >= 0.3) & (u_flat <= 0.7)
            mask_late = u_flat > 0.7

            loss_early = loss_unreduced[mask_early].mean().item() if mask_early.any() else 0.0
            loss_mid = loss_unreduced[mask_mid].mean().item() if mask_mid.any() else 0.0
            loss_late = loss_unreduced[mask_late].mean().item() if mask_late.any() else 0.0

        # --- Normalizer Stats ---
        batch_mean = raw_latents.mean()
        batch_var = raw_latents.float().var(dim=0, unbiased=False).max()
        
        global_mean = 0.0
        if hasattr(self.normalizer, "mean") and torch.is_tensor(self.normalizer.mean):
            global_mean = self.normalizer.get_layer_stat(self.normalizer.mean, layer_idx).mean()
            
        global_var = 1.0
        if hasattr(self.normalizer, "var") and torch.is_tensor(self.normalizer.var):
            global_var = self.normalizer.get_layer_stat(self.normalizer.var, layer_idx).max()

        return SimpleNamespace(
            latents=outputs,
            timesteps=timesteps,
            loss=loss,
            loss_raw=loss_raw,
            tail_weighted_mse=tail_weighted_mse,
            tail_dim_fraction=tail_dim_fraction,
            tail_weight_mean=tail_weight_mean,
            tail_weight_max=tail_weight_max,
            tgt_norm=tgt_norm.mean(),
            latent_pre_l2=latent_pre_l2.mean(),
            latent_post_l2=latent_post_l2.mean(),
            pre_l2_std=pre_l2_std,
            post_l2_std=post_l2_std,
            latent_pre_l1=latent_pre_l1.mean(),
            latent_post_l1=latent_post_l1.mean(),
            loss_rel=loss_rel,
            cos_sim=cos_sim,
            tail_region_mse=tail_region_mse,
            non_tail_region_mse=non_tail_region_mse,
            loss_early=loss_early,
            loss_mid=loss_mid,
            loss_late=loss_late,
            batch_mean=batch_mean,
            batch_var=batch_var,
            global_mean=global_mean,
            global_var=global_var,
        )

def load_glp(weights_folder, device="cuda:0", checkpoint="final", local_files_only=False):
    """
    Load GLP from either:
    - local folder path
    - Hugging Face repo id (auto-downloaded via snapshot_download)

    The checkpoint can be:
    - "final" (loads final.safetensors)
    - a milestone folder name under the root (e.g. "100M")

    local_files_only behavior:
    - True: only use local HF cache; fail if missing
    - False: allow network download
    - None (default): try local cache first, then fall back to network
    """
    resolved_folder = Path(weights_folder).expanduser()

    # If a local folder is provided, use it directly.
    # Otherwise treat input as a Hub repo id and download it.
    if not resolved_folder.exists():
        if checkpoint == "final":
            allow_patterns = ["config.yaml", "rep_statistics.pt", "final.safetensors"]
        else:
            allow_patterns = [
                "config.yaml",
                "rep_statistics.pt",
                f"{checkpoint}.safetensors",
                f"{checkpoint}/config.yaml",
                f"{checkpoint}/rep_statistics.pt",
                f"{checkpoint}/final.safetensors",
            ]

        download_kwargs = {
            "repo_id": weights_folder,
            "allow_patterns": allow_patterns,
        }

        if local_files_only is True:
            local_dir = snapshot_download(local_files_only=True, **download_kwargs)
        elif local_files_only is False:
            local_dir = snapshot_download(local_files_only=False, **download_kwargs)

        resolved_folder = Path(local_dir)

    # Allow loading a milestone subfolder when checkpoint names a directory.
    # Example: root/100M/{config.yaml, rep_statistics.pt, final.safetensors}
    checkpoint_dir = resolved_folder / checkpoint
    if checkpoint != "final" and checkpoint_dir.is_dir():
        resolved_folder = checkpoint_dir
        checkpoint = "final"

    config = OmegaConf.load(str(resolved_folder / "config.yaml"))
    rep_stats_file = resolved_folder / "rep_statistics.pt"
    rep_stats_path = str(rep_stats_file)

    normalizer_config = None
    if "glp_kwargs" in config and "normalizer_config" in config.glp_kwargs:
        normalizer_config = config.glp_kwargs.normalizer_config

    normalization_method = "gaussian"
    if normalizer_config is not None and "normalization_method" in normalizer_config:
        normalization_method = _canonicalize_normalization_method(
            normalizer_config.normalization_method
        )

    # Rewrite rep_statistic to the resolved local path when appropriate.
    if normalizer_config is not None:
        if _normalization_requires_stats(normalization_method):
            if not rep_stats_file.exists():
                raise FileNotFoundError(
                    f"Missing required normalization statistics at {rep_stats_file} "
                    f"for method '{normalization_method}'"
                )
            normalizer_config.rep_statistic = rep_stats_path
        else:
            # log_norm does not require running mean/var-like stats.
            normalizer_config.rep_statistic = rep_stats_path if rep_stats_file.exists() else ""
    # Fallback for older/alternate config shapes.
    elif "rep_statistic" in config:
        if rep_stats_file.exists():
            config.rep_statistic = rep_stats_path

    if "glp_kwargs" in config and "sampling_method" in config.glp_kwargs:
        config.glp_kwargs.noise_sampling_method = config.glp_kwargs.sampling_method
        del config.glp_kwargs["sampling_method"]

    OmegaConf.resolve(config)
    model = GLP(**config.glp_kwargs)
    model.load_pretrained(resolved_folder, name=checkpoint)
    if "glp_kwargs" in config and "solver" in config.glp_kwargs:
        model.set_inference_solver(config.glp_kwargs.solver)
    model.to(device)
    return model, model.normalizer.mean, model.normalizer.var
