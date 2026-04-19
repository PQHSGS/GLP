import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# Utility: Sinusoidal Embedding
# -----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: (B,) tensor of timesteps
        Returns: (B, dim) tensor of embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -----------------------------
# Main Classifier
# -----------------------------
class ConceptClassifier(nn.Module):
    def __init__(
        self,
        d_input,
        d_model=256,
        d_mlp=512,
        n_layers=4,
        cond_dim=128,      
    ):
        super().__init__()

        # 1. High-frequency sinusoidal mapping for t
        self.time_embed = SinusoidalPosEmb(cond_dim)

        # 2. MLP to process the embedded time
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        # Input projection
        self.in_proj = nn.Linear(d_input, d_model)

        # Residual blocks
        self.layers = nn.ModuleList([
            ClassifierMLPBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                cond_dim=cond_dim,
            )
            for _ in range(n_layers)
        ])

        # Output head
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, z_t, t):
        # -------------------------
        # Compute conditioning features
        # -------------------------
        # Pass raw t (B,) through sinusoidal -> (B, cond_dim)
        t_freq = self.time_embed(t) 
        cond_emb = self.cond_mlp(t_freq) 

        # -------------------------
        # Input
        # -------------------------
        x = self.in_proj(z_t)

        # -------------------------
        # Residual blocks
        # -------------------------
        for layer in self.layers:
            x = layer(x, cond_emb)

        # -------------------------
        # Output
        # -------------------------
        x = self.out_norm(x)
        logit = self.out_proj(x).squeeze(-1)

        return logit

    def log_prob(self, z_t, t):
        return F.logsigmoid(self.forward(z_t, t))

# -----------------------------
# Residual Block with FiLM
# -----------------------------
class ClassifierMLPBlock(nn.Module):
    """
    MLP block with:
    - Pre-norm
    - SwiGLU
    - Full FiLM conditioning (scale + shift)
    """

    def __init__(self, d_model, d_mlp, cond_dim):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        # SwiGLU
        self.up_proj   = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.act = nn.SiLU()

        # FiLM conditioning
        self.scale_proj = nn.Linear(cond_dim, d_model)
        self.shift_proj = nn.Linear(cond_dim, d_model)

    def forward(self, x, cond_emb):
        resid = x
        x = self.norm(x)

        # FiLM modulation
        scale = self.scale_proj(cond_emb)
        shift = self.shift_proj(cond_emb)
        x = (1 + scale) * x + shift

        # SwiGLU
        gate = self.gate_proj(x)
        up   = self.up_proj(x)
        x = self.act(gate) * up
        x = self.down_proj(x)

        return x + resid