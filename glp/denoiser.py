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

    if method in {"gaussian", "log_norm", "rmsnorm"}:
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
        "Expected one of ['gaussian', 'log_norm', 'rmsnorm', 'quantile_XX', '99', '0.99']."
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

        if self.normalization_method == "gaussian" or self.normalization_method.startswith("quantile_"):
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

        if self.normalization_method == "gaussian" or self.normalization_method.startswith("quantile_"):
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

# ==========================
#     Denoiser Classes
# ==========================
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

class TransformerMLPDenoiser(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_mlp=1536,
        d_input=1536,
        n_layers=12,
        multi_layer_n_layers=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_input = d_input
        self.n_layers = n_layers
        self.multi_layer_n_layers = multi_layer_n_layers

        self.layers = nn.ModuleList([
            TransformerMLPBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                d_input=d_input
            ) for _ in range(n_layers)
        ])
        self.in_proj = nn.Linear(d_input, d_model)
        self.out_proj = nn.Linear(d_model, d_input)

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
    def __init__(self, normalizer_config, denoiser_config, tracedict_config=None):
        super().__init__()
        self.normalizer = Normalizer.from_config(**normalizer_config)
        self.denoiser = Denoiser(**denoiser_config)
        self.scheduler = flow_matching.fm_scheduler()
        self.tracedict_config = tracedict_config

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

        phase = 2
        # Legacy two-phase schedule. Disabled while returning to the baseline
        # Flow Matching setup: plain MSE with uniformly sampled u.
        # if two_phase and global_step is not None and total_steps is not None:
        #     if global_step <= 0.4 * total_steps:
        #         phase = 1
        #
        # if phase == 1:
        #     if u is None:
        #         u_normal = torch.randn(latents.shape[0], device=latents.device, generator=generator)
        #         u = torch.sigmoid(u_normal)
        # else:
        #     if u is None:
        #         u = torch.rand(latents.shape[0], device=latents.device, generator=generator)
        if u is None:
            u = torch.rand(latents.shape[0], device=latents.device, generator=generator)

        # prepare flow matching inputs and target
        noise = torch.randn(latents.shape, dtype=latents.dtype, generator=generator).to(latents.device)
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

        # compute loss: baseline MSE, optionally reweighted on rare large
        # clean-latent coordinates for tail-aware experiments.
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        mse_element = torch.nn.functional.mse_loss(
            outputs_f32,
            target_f32,
            reduction='none',
        )
        if self.normalizer.normalization_method == "log_norm":
            mse_element = mse_element * 1e5

        loss_unreduced = mse_element.view(latents.shape[0], -1).mean(dim=-1)
        tail_base_mse = loss_unreduced.detach().mean()
        tail_aware_weight = float(loss_kwargs.get("tail_aware_weight", 0.0) or 0.0)
        tail_aware_min_weight = float(loss_kwargs.get("tail_aware_min_weight", 0.1) or 0.0)
        tail_aware_max_weight = float(loss_kwargs.get("tail_aware_max_weight", 10.0) or 0.0)

        tail_fraction = outputs_f32.new_tensor(0.0)
        tail_weight_mean = outputs_f32.new_tensor(1.0)
        tail_weight_max = outputs_f32.new_tensor(1.0)
        tail_region_mse = outputs_f32.new_tensor(0.0)
        non_tail_region_mse = outputs_f32.new_tensor(0.0)

        if tail_aware_weight > 0.0:
            with torch.no_grad():
                raw_target = self.normalizer.denormalize(latents, layer_idx=layer_idx).detach().float()
                raw_magnitude = raw_target.abs()
                batch_mean_mag = raw_magnitude.mean(dim=-1, keepdim=True).clamp_min(1e-8)
                tail_multiplier = (raw_magnitude / batch_mean_mag).pow(tail_aware_weight)
                if tail_aware_min_weight > 0.0 or tail_aware_max_weight > 0.0:
                    clamp_kwargs = {}
                    if tail_aware_min_weight > 0.0:
                        clamp_kwargs["min"] = tail_aware_min_weight
                    if tail_aware_max_weight > 0.0:
                        clamp_kwargs["max"] = tail_aware_max_weight
                    tail_multiplier = tail_multiplier.clamp(**clamp_kwargs)
                tail_mask = tail_multiplier > 1.0
                tail_fraction = tail_mask.float().mean()
                tail_weight_mean = tail_multiplier.mean()
                tail_weight_max = tail_multiplier.max()
                if tail_mask.any():
                    tail_region_mse = mse_element.detach()[tail_mask].mean()
                non_tail_mask = ~tail_mask
                if non_tail_mask.any():
                    non_tail_region_mse = mse_element.detach()[non_tail_mask].mean()
            loss_unreduced = (mse_element * tail_multiplier.to(mse_element.dtype)).view(latents.shape[0], -1).mean(dim=-1)
        loss = loss_unreduced.mean()
        tail_weighted_mse = loss.detach()

        # Legacy phase-2 pseudo-Huber objective. Kept for future experiments.
        # u_t = meta["u"].to(device=outputs.device, dtype=torch.float32).view(-1, 1, 1)
        # w = 1.0 / (1.0 - u_t + 1e-4)
        # error = outputs_f32 - target_f32
        # delta = 0.01
        # loss_raw_hub = (delta ** 2) * (torch.sqrt(1.0 + (error / delta) ** 2) - 1.0)
        # loss_unreduced = (w * loss_raw_hub).view(latents.shape[0], -1).mean(dim=-1)
        # loss = loss_unreduced.mean()

        # ===== proper metrics =====
        raw_latents = self.normalizer.denormalize(latents, layer_idx=layer_idx).view(-1, latents.shape[-1])

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
        weights = 1.0 / tgt_norm
        tgt_norm_sq = (tgt ** 2).sum(dim=-1) + 1e-8
        loss_rel = ((pred - tgt) ** 2).sum(dim=-1) / tgt_norm_sq
        loss_rel = loss_rel.mean()

        # map back to raw space
        pred_raw = self.normalizer.denormalize(pred, layer_idx)
        tgt_raw  = self.normalizer.denormalize(tgt, layer_idx)

        # raw-space MSE (THIS is comparable across normalization)
        loss_raw = torch.nn.functional.mse_loss(pred_raw, tgt_raw)

        # cosine similarity (KEEP)
        cos_sim = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1).mean()

        # --- Legacy diagnostic metrics (disabled) ---
        # These 5-step manifold/sparsity diagnostics are intentionally left in
        # the file but disabled while simplifying training back to MSE + AdamW.
        calc_metrics = False
        # if global_step is None or (global_step + 1) % 5 == 0:
        #     calc_metrics = True

        PR, H_SVD, kappa, k_99 = 0.0, 0.0, 0.0, 0.0
        dead_ratio, hoyer_sparsity = 0.0, 0.0
        loss_early, loss_mid, loss_late = 0.0, 0.0, 0.0
        
        if False and calc_metrics:
            with torch.no_grad():
                # --- Timestep Loss Mask ---
                # Calculate raw loss per batch item
                u_flat = meta["u"].view(-1).to(device=pred.device)
                
                mask_early = u_flat < 0.3
                mask_mid = (u_flat >= 0.3) & (u_flat <= 0.7)
                mask_late = u_flat > 0.7
                
                loss_early = loss_unreduced[mask_early].mean().item() if mask_early.any() else 0.0
                loss_mid = loss_unreduced[mask_mid].mean().item() if mask_mid.any() else 0.0
                loss_late = loss_unreduced[mask_late].mean().item() if mask_late.any() else 0.0

                # --- Manifold Spectral Measurements ---
                X = latents.view(-1, latents.shape[-1]).float()
                
                X_centered = X - X.mean(dim=0)
                s = torch.linalg.svdvals(X_centered)
                
                # Robust Noise Floor Handling
                s_max = s[0]
                s_clean = s[s > (s_max * 1e-6)] 
                lambdas = s_clean ** 2
                sum_lambdas = lambdas.sum()
                
                # 1. Participation Ratio (PR)
                PR = (sum_lambdas ** 2) / (lambdas ** 2).sum()
                
                # 2. Spectral Entropy (H_SVD)
                p = lambdas / (sum_lambdas + 1e-9)
                H_SVD = -torch.sum(p * torch.log(p + 1e-9))
                
                # 3. Truncated Condition Number (kappa)
                ref_idx = min(63, len(s) - 1)
                kappa = s[0] / (s[ref_idx] + 1e-9)
                
                # 4. Dimension for 99% Variance (k_99)
                cum_var = torch.cumsum(lambdas, dim=0) / sum_lambdas
                k_99 = (cum_var < 0.99).sum().float()

                # --- Polysemantic/Sparsity Measurements ---
                X_raw = raw_latents.float()
                D = X_raw.shape[-1]
                
                # 1. Dead Ratio (What % is functionally zero?)
                is_active = (X_raw.abs() > 1e-3).float()
                active_ratio = is_active.mean()
                dead_ratio = 1.0 - active_ratio
                
                # 2. Hoyer Sparsity (Scale-Invariant Concentration)
                l1_norm = X_raw.abs().sum(dim=-1)
                l2_norm = torch.linalg.vector_norm(X_raw, dim=-1)
                
                sqrt_d = math.sqrt(D)
                hoyer = (sqrt_d - (l1_norm / (l2_norm + 1e-9))) / (sqrt_d - 1.0)
                hoyer_sparsity = hoyer.mean()

        # --- Normalizer Stats ---
        batch_mean = latents.mean().item()
        batch_var = latents.var().item()
        
        global_mean = 0.0
        if hasattr(self.normalizer, "mean") and torch.is_tensor(self.normalizer.mean):
            global_mean = self.normalizer.mean.mean().item()
            
        global_var = 1.0
        if hasattr(self.normalizer, "var") and torch.is_tensor(self.normalizer.var):
            global_var = self.normalizer.var.var().item()

        return SimpleNamespace(
            latents=outputs,
            timesteps=timesteps,
            loss=loss,
            tgt_norm=tgt_norm.mean(),
            latent_pre_l2=latent_pre_l2.mean(),
            latent_post_l2=latent_post_l2.mean(),
            pre_l2_std=pre_l2_std,
            post_l2_std=post_l2_std,
            latent_pre_l1=latent_pre_l1.mean(),
            latent_post_l1=latent_post_l1.mean(),
            loss_rel=loss_rel,
            loss_raw=loss_raw,
            cos_sim=cos_sim,
            tail_aware_weight=tail_aware_weight,
            tail_aware_min_weight=tail_aware_min_weight,
            tail_aware_max_weight=tail_aware_max_weight,
            tail_fraction=tail_fraction,
            tail_weight_mean=tail_weight_mean,
            tail_weight_max=tail_weight_max,
            tail_base_mse=tail_base_mse,
            tail_weighted_mse=tail_weighted_mse,
            tail_region_mse=tail_region_mse,
            non_tail_region_mse=non_tail_region_mse,
            PR=PR,
            H_SVD=H_SVD,
            kappa=kappa,
            k_99=k_99,
            dead_ratio=dead_ratio,
            hoyer_sparsity=hoyer_sparsity,
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
    OmegaConf.resolve(config)
    model = GLP(**config.glp_kwargs)
    model.load_pretrained(resolved_folder, name=checkpoint)
    model.to(device)
    return model, model.normalizer.mean, model.normalizer.var
