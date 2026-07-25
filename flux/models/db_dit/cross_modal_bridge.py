"""Cross-Modal Bridge with Gated Attention (CBGA).

Implements bidirectional cross-attention between vision and audio branches
with learnable gates initialized to zero. This ensures stable training:
the bridge doesn't disrupt pretrained single-modality branches at the start.

Following Seedance 2.0's described CBGA mechanism.
"""

import torch
import torch.nn as nn

from flux.models.db_dit.attention import MultiHeadAttention


class CBGABlock(nn.Module):
    """Cross-Branch Gated Attention block.

    Performs:
    1. Audio tokens attend to video tokens (with learnable gate)
    2. Video tokens attend to audio tokens (with learnable gate)

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Video -> Audio cross-attention
        # Audio tokens query video tokens
        self.v2a_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout, cross_attn=True
        )
        self.v2a_norm = nn.LayerNorm(dim)
        # Gate: scalar parameter initialized to 0
        self.v2a_gate = nn.Parameter(torch.zeros(1))

        # Audio -> Video cross-attention
        # Video tokens query audio tokens
        self.a2v_attn = MultiHeadAttention(
            dim, num_heads, qk_norm=qk_norm, dropout=dropout, cross_attn=True
        )
        self.a2v_norm = nn.LayerNorm(dim)
        # Gate: scalar parameter initialized to 0
        self.a2v_gate = nn.Parameter(torch.zeros(1))

        # Optional timestep-dependent gate modulation
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2),  # One gate per direction
        )
        # Zero-init the projection
        nn.init.zeros_(self.t_proj[1].weight)
        nn.init.zeros_(self.t_proj[1].bias)

        # Warmup tracking
        self.register_buffer("warmup_steps", torch.tensor(50000))
        self.register_buffer("current_step", torch.tensor(0))

    def set_step(self, step: int):
        """Update current training step for gate warmup scheduling."""
        self.current_step.fill_(step)

    def get_gate_scale(self) -> float:
        """Get current gate scale based on warmup schedule.

        Returns a multiplier in [0, 1] that linearly increases from 0 to 1
        over warmup_steps.
        """
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self.current_step.item() / self.warmup_steps.item())

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CBGA cross-modal attention.

        Args:
            v_tokens: Video tokens (B, N_v, D).
            a_tokens: Audio tokens (B, N_a, D).
            t_emb: Timestep embedding (B, D) for gate modulation.

        Returns:
            Tuple of (updated_v_tokens, updated_a_tokens).
        """
        warmup_scale = self.get_gate_scale()

        # Timestep-dependent gate modulation
        t_gates = torch.sigmoid(self.t_proj(t_emb))  # (B, 2)
        t_gate_v2a = t_gates[:, 0:1]  # (B, 1)
        t_gate_a2v = t_gates[:, 1:2]  # (B, 1)

        # 1. Video -> Audio: audio tokens query video tokens
        a_norm = self.v2a_norm(a_tokens)
        attn_v2a = self.v2a_attn(a_norm, context=v_tokens)
        gate_v2a = warmup_scale * self.v2a_gate * t_gate_v2a.unsqueeze(-1)
        a_tokens = a_tokens + gate_v2a * attn_v2a

        # 2. Audio -> Video: video tokens query audio tokens
        v_norm = self.a2v_norm(v_tokens)
        attn_a2v = self.a2v_attn(v_norm, context=a_tokens)
        gate_a2v = warmup_scale * self.a2v_gate * t_gate_a2v.unsqueeze(-1)
        v_tokens = v_tokens + gate_a2v * attn_a2v

        return v_tokens, a_tokens
