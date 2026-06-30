"""Dual-Branch Transformer Block.

Orchestrates the vision branch, audio branch, and cross-modal bridge (CBGA)
within a single transformer layer.

Each layer executes:
  1. Vision Branch (STDiT): spatial + temporal + cross-text + FFN
  2. Audio Branch (DiT): self + cross-text + FFN
  3. CBGA (if layer is in cbga_layers): bidirectional vision-audio attention
"""

import torch
import torch.nn as nn

from seedance.models.db_dit.vision_branch import VisionBranchBlock
from seedance.models.db_dit.audio_branch import AudioBranchBlock
from seedance.models.db_dit.cross_modal_bridge import CBGABlock


class DualBranchBlock(nn.Module):
    """A single dual-branch transformer layer.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        cond_dim: Timestep conditioning dimension.
        ffn_ratio: FFN hidden dimension ratio.
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
        layer_idx: Index of this layer (0-indexed).
        cbga_layers: Set of layer indices where CBGA is applied.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        ffn_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        layer_idx: int = 0,
        cbga_layers: set[int] | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.has_cbga = cbga_layers is not None and layer_idx in cbga_layers

        # Vision branch (STDiT)
        self.vision_block = VisionBranchBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            ffn_ratio=ffn_ratio,
            qk_norm=qk_norm,
            dropout=dropout,
        )

        # Audio branch (DiT)
        self.audio_block = AudioBranchBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            ffn_ratio=ffn_ratio,
            qk_norm=qk_norm,
            dropout=dropout,
        )

        # Cross-modal bridge (CBGA) — only at specified layers
        if self.has_cbga:
            self.cbga = CBGABlock(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                dropout=dropout,
            )
        else:
            self.cbga = None

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        video_grid: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one dual-branch block.

        Args:
            v_tokens: Video tokens (B, N_v, D).
            a_tokens: Audio tokens (B, N_a, D).
            t_emb: Timestep embedding (B, cond_dim).
            text_emb: Text embeddings (B, L_text, D).
            video_grid: (T_v, H_v, W_v) grid dimensions.

        Returns:
            Tuple of (updated_v_tokens, updated_a_tokens).
        """
        # 1. Vision branch forward
        v_tokens = self.vision_block(v_tokens, t_emb, text_emb, video_grid)

        # 2. Audio branch forward
        a_tokens = self.audio_block(a_tokens, t_emb, text_emb)

        # 3. Cross-modal bridge (if enabled for this layer)
        if self.cbga is not None:
            v_tokens, a_tokens = self.cbga(v_tokens, a_tokens, t_emb)

        return v_tokens, a_tokens
