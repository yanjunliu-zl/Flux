"""Dual-Branch Transformer Block.

Orchestrates the vision branch, audio branch, cross-modal bridge (CBGA),
and optional lip-sync bridge within a single transformer layer.

Each layer executes:
  1. Vision Branch (STDiT): spatial + temporal + cross-text + FFN
  2. Audio Branch (DiT): self + cross-text + FFN
  3. CBGA (if layer is in cbga_layers): bidirectional vision-audio attention
  4. LipSyncBridge (if layer is in lip_sync_layers): mouth-focused audio attention
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional

from flux.models.db_dit.vision_branch import VisionBranchBlock
from flux.models.db_dit.audio_branch import AudioBranchBlock
from flux.models.db_dit.cross_modal_bridge import CBGABlock


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
        lip_sync_layers: Set of layer indices where LipSyncBridge is applied.
            Default: same as cbga_layers. Set to empty set to disable.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        context_dim: int | None = None,
        ffn_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        layer_idx: int = 0,
        cbga_layers: set[int] | None = None,
        lip_sync_layers: set[int] | None = None,
        moe_config: dict | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._grad_ckpt = False  # Set by distributed.py for single-GPU activation checkpointing
        self.has_cbga = cbga_layers is not None and layer_idx in cbga_layers
        self.has_lip_sync = (
            lip_sync_layers is not None
            and layer_idx in lip_sync_layers
        )

        # Vision branch (STDiT) — cross-attention matches text encoder dim
        self.vision_block = VisionBranchBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            context_dim=context_dim,
            ffn_ratio=ffn_ratio,
            qk_norm=qk_norm,
            dropout=dropout,
            moe_config=moe_config,
        )

        # Audio branch (DiT) — cross-attention matches text encoder dim
        self.audio_block = AudioBranchBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            context_dim=context_dim,
            ffn_ratio=ffn_ratio,
            qk_norm=qk_norm,
            dropout=dropout,
            moe_config=moe_config,
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

        # Lip-sync bridge — mouth-focused cross-attention
        if self.has_lip_sync:
            from flux.models.mouth_roi_attention import LipSyncBridge
            self.lip_sync = LipSyncBridge(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                dropout=dropout,
            )
        else:
            self.lip_sync = None

    def set_step(self, step: int):
        """Update step count for warmup scheduling in CBGA and LipSync."""
        if self.cbga is not None:
            self.cbga.set_step(step)
        if self.lip_sync is not None:
            self.lip_sync.set_step(step)

    def _forward_impl(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        video_grid: tuple[int, int, int],
        mouth_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """Actual forward logic (called directly or via checkpoint)."""
        T_v, H_v, W_v = video_grid
        moe_v, moe_a = None, None

        v_result = self.vision_block(v_tokens, t_emb, text_emb, video_grid)
        if isinstance(v_result, tuple):
            v_tokens, moe_v = v_result
        else:
            v_tokens = v_result

        a_result = self.audio_block(a_tokens, t_emb, text_emb)
        if isinstance(a_result, tuple):
            a_tokens, moe_a = a_result
        else:
            a_tokens = a_result

        if self.cbga is not None:
            v_tokens, a_tokens = self.cbga(v_tokens, a_tokens, t_emb)

        if self.lip_sync is not None:
            v_tokens = self.lip_sync(
                v_tokens, a_tokens, mouth_mask,
                T_lat=T_v, H_lat=H_v, W_lat=W_v,
            )

        moe_aux = None
        if moe_v is not None or moe_a is not None:
            moe_aux = {}
            for src in [moe_v, moe_a]:
                if src:
                    for k, v in src.items():
                        moe_aux[k] = moe_aux.get(k, 0.0) + v

        return v_tokens, a_tokens, moe_aux

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        text_emb: torch.Tensor,
        video_grid: tuple[int, int, int],
        mouth_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one dual-branch block.

        Args:
            v_tokens: Video tokens (B, N_v, D).
            a_tokens: Audio tokens (B, N_a, D).
            t_emb: Timestep embedding (B, cond_dim).
            text_emb: Text embeddings (B, L_text, D).
            video_grid: (T_v, H_v, W_v) grid dimensions.
            mouth_mask: Optional spatial mask (B, H_v, W_v) for lip-sync attention.

        Returns:
            Tuple of (updated_v_tokens, updated_a_tokens, moe_aux_losses).
        """
        if self._grad_ckpt and self.training:
            # Gradient checkpointing: recompute activations in backward instead of storing them
            # Saves ~90% activation VRAM at cost of ~20% extra compute
            return checkpoint(
                self._forward_impl,
                v_tokens, a_tokens, t_emb, text_emb, video_grid, mouth_mask,
                use_reentrant=False,
            )
        return self._forward_impl(v_tokens, a_tokens, t_emb, text_emb, video_grid, mouth_mask)

    def get_viseme_logits(
        self, a_tokens: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get viseme classification logits from LipSyncBridge.

        Args:
            a_tokens: Audio tokens (B, N_a, D).

        Returns:
            Viseme logits (B, N_a, num_visemes) or None if no lip-sync layer.
        """
        if self.lip_sync is not None:
            return self.lip_sync.compute_viseme_logits(a_tokens)
        return None
