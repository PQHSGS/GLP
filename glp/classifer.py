# glp/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from glp.denoiser import timestep_embedding


class ConceptClassifier(nn.Module):
    """
    Classifier p_φ(y=1 | z_t, t) cho Classifier Guidance.
    
    Design choices:
    - Input: noisy activation z_t (d_input dim) + timestep t
    - Kiến trúc: MLP với residual connections
    - Tương tự TransformerMLPBlock của GLP nhưng nhỏ hơn
    - Output: scalar logit → sigmoid → probability
    

    """
    def __init__(
        self,
        d_input,          # activation dim: 2048 (Llama1B) or 4096 (Llama8B)
        d_model=256,      # hidden dim — nhỏ hơn GLP để train nhanh
        d_mlp=512,        # MLP expansion
        n_layers=4,       # đủ để nonlinear, không quá sâu
        t_embed_dim=256,  # timestep embedding dim
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.t_embed_dim = t_embed_dim

        # 1. Timestep embedding (giống GLP gốc)
        self.time_embed = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # 2. Input projection: z_t (d_input) → d_model
        self.in_proj = nn.Linear(d_input, d_model)

        # 3. MLP blocks với residual + timestep conditioning
        self.layers = nn.ModuleList([
            ClassifierMLPBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                t_embed_dim=t_embed_dim,
            )
            for _ in range(n_layers)
        ])

        # 4. Output head: d_model → 1 (binary classification)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, z_t, t):
        """
        Args:
            z_t: (B, d_input) — noisy activation tại timestep t
            t:   (B,)         — timestep values trong [0, 1]
        Returns:
            logits: (B,) — raw logit (chưa sigmoid)
        """
        # timestep embedding
        t_emb = timestep_embedding(
            t.flatten(),
            self.t_embed_dim
        ).to(z_t.dtype)                          # (B, t_embed_dim)
        t_emb = self.time_embed(t_emb)           # (B, t_embed_dim)

        # project input
        x = self.in_proj(z_t)                    # (B, d_model)

        # MLP blocks
        for layer in self.layers:
            x = layer(x, t_emb)                  # (B, d_model)

        # output
        x = self.out_norm(x)                     # (B, d_model)
        logit = self.out_proj(x).squeeze(-1)     # (B,)
        return logit

    def log_prob(self, z_t, t):
        """Convenience function để dùng trong guidance."""
        return F.logsigmoid(self.forward(z_t, t))


class ClassifierMLPBlock(nn.Module):
    """
    MLP block cho classifier.
    Giống TransformerMLPBlock của GLP nhưng:
    - Không có layer_idx conditioning (không cần)
    - LayerNorm trước (pre-norm) thay vì sau
    - Timestep conditioning qua multiplicative gate
      (giống GLP gốc để consistent)
    """
    def __init__(self, d_model, d_mlp, t_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # SwiGLU (giống GLP)
        self.up_proj   = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.act = nn.SiLU()

        # timestep conditioning (giống GLP)
        self.time_proj = nn.Linear(t_embed_dim, d_mlp)

    def forward(self, x, t_emb):
        """
        x:     (B, d_model)
        t_emb: (B, t_embed_dim)
        """
        resid = x
        x = self.norm(x)

        # SwiGLU với timestep modulation
        gate = self.gate_proj(x)
        t    = self.time_proj(t_emb)
        gate = gate * t                  # multiplicative conditioning
        x    = self.act(gate) * self.up_proj(x)
        x    = self.down_proj(x)

        return x + resid                 # residual connection