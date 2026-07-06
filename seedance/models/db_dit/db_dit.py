"""DB-DiT (Dual-Branch Diffusion Transformer) — Full Model.

The core of Seedance 2.0: a dual-branch transformer that jointly generates
video and audio latents through a shared diffusion process.

Architecture:
  - Vision branch: N STDiT blocks with spatial + temporal attention
  - Audio branch: N DiT blocks with self + cross-attention
  - Cross-modal bridge (CBGA): bidirectional gated cross-attention at specific layers
  - Flow matching velocity prediction for both modalities
"""

import torch
import torch.nn as nn

from seedance.models.common.embedding import PatchEmbed, TimestepEmbedding
from seedance.models.db_dit.dual_branch_block import DualBranchBlock
from seedance.models.db_dit.mm_rope import MMRoPE


class DBDiT(nn.Module):
    """Dual-Branch Diffusion Transformer for joint audio-video generation.

    Args:
        dim: Hidden dimension.
        num_layers: Number of dual-branch transformer layers.
        num_heads: Number of attention heads.
        ffn_ratio: FFN hidden dimension ratio.
        qk_norm: Whether to apply QK normalization.
        dropout: Dropout rate.
        cbga_layers: Layer indices (0-indexed) where CBGA is applied.
        cbga_gate_warmup_steps: Steps to linearly warm up CBGA gates.
        lip_sync_layers: Layer indices where LipSyncBridge is applied.
            Default: same as cbga_layers. Empty list to disable.
        video_patch_size: Video patch size (T, H, W).
        video_latent_channels: VideoVAE latent channels (16).
        video_rope_theta: MM-RoPE theta for video.
        video_rope_dim_t: RoPE dimensions for temporal axis.
        video_rope_dim_h: RoPE dimensions for height axis.
        video_rope_dim_w: RoPE dimensions for width axis.
        audio_patch_size: Audio patch size (F, T).
        audio_latent_channels: AudioVAE latent channels (8).
        audio_rope_theta: RoPE theta for audio.
    """

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        context_dim: int | None = None,
        ffn_ratio: float = 4.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        cbga_layers: list[int] | None = None,
        cbga_gate_warmup_steps: int = 50000,
        lip_sync_layers: list[int] | None = None,
        moe_config: dict | None = None,
        video_patch_size: tuple[int, int, int] = (1, 2, 2),
        video_latent_channels: int = 16,
        video_rope_theta: float = 10000.0,
        video_rope_dim_t: int | None = None,
        video_rope_dim_h: int | None = None,
        video_rope_dim_w: int | None = None,
        audio_patch_size: tuple[int, int] = (1, 4),
        audio_latent_channels: int = 8,
        audio_rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.cbga_layers = set(cbga_layers or [])
        # Lip-sync layers default to same as CBGA layers
        self.lip_sync_layers = set(
            lip_sync_layers if lip_sync_layers is not None
            else (cbga_layers or [])
        )

        # Timestep embedding
        self.t_embed = TimestepEmbedding(dim)

        # Video patch embedding
        self.video_patch_embed = PatchEmbed(
            in_channels=video_latent_channels,
            embed_dim=dim,
            patch_size=video_patch_size,
        )

        # Audio patch embedding
        self.audio_patch_embed = PatchEmbed(
            in_channels=audio_latent_channels,
            embed_dim=dim,
            patch_size=audio_patch_size,
        )

        # MM-RoPE for video (3D)
        self.video_rope = MMRoPE(
            dim=dim // num_heads,  # Per-head dimension
            rope_dim_t=video_rope_dim_t,
            rope_dim_h=video_rope_dim_h,
            rope_dim_w=video_rope_dim_w,
            theta=video_rope_theta,
        )

        # RoPE for audio (1D)
        from seedance.models.db_dit.mm_rope import MMRoPE as AudioRoPE
        self.audio_rope = AudioRoPE(
            dim=dim // num_heads,
            rope_dim_a=dim // num_heads,
            theta=audio_rope_theta,
        )

        # Dual-branch transformer layers
        self.layers = nn.ModuleList([
            DualBranchBlock(
                dim=dim,
                num_heads=num_heads,
                cond_dim=dim,
                context_dim=context_dim,
                ffn_ratio=ffn_ratio,
                qk_norm=qk_norm,
                dropout=dropout,
                layer_idx=i,
                cbga_layers=self.cbga_layers,
                lip_sync_layers=self.lip_sync_layers,
                moe_config=moe_config,
            )
            for i in range(num_layers)
        ])

        # Final LayerNorm
        self.final_norm_video = nn.LayerNorm(dim)
        self.final_norm_audio = nn.LayerNorm(dim)

        # Output heads: project back to latent space
        # Video: patches of size (1, 2, 2) -> 16 * 1 * 2 * 2 = 64 output channels
        video_patch_out = video_latent_channels * video_patch_size[0] * video_patch_size[1] * video_patch_size[2]
        self.video_head = nn.Linear(dim, video_patch_out)

        # Audio: patches of size (1, 4) -> 8 * 1 * 4 = 32 output channels
        audio_patch_out = audio_latent_channels * audio_patch_size[0] * audio_patch_size[1]
        self.audio_head = nn.Linear(dim, audio_patch_out)

        # Initialize output heads to zero for stable training
        nn.init.zeros_(self.video_head.weight)
        nn.init.zeros_(self.video_head.bias)
        nn.init.zeros_(self.audio_head.weight)
        nn.init.zeros_(self.audio_head.bias)

    def set_step(self, step: int):
        """Update training step for CBGA and LipSync gate warmup."""
        for layer in self.layers:
            if hasattr(layer, "set_step"):
                layer.set_step(step)
            elif layer.cbga is not None:
                layer.cbga.set_step(step)

    def forward(
        self,
        v_latent: torch.Tensor,
        a_latent: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        first_frame_mask: torch.Tensor | None = None,
        mouth_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting velocity field for flow matching.

        Args:
            v_latent: Noisy video latent (B, C_v, T_v, H_v, W_v).
            a_latent: Noisy audio latent (B, C_a, F_a, T_a).
            t: Timestep (B,) in [0, 1].
            text_emb: Text embeddings (B, L_text, D_text).
            first_frame_mask: Optional mask (B, 1, 1, H_v, W_v) for I2VA.
                              First frame is clean (t=1), model predicts 0 velocity there.
            mouth_mask: Optional spatial mask (B, H_v, W_v) for lip-sync attention.
                        Higher values = mouth region. Can be 0-filled for non-talking samples.

        Returns:
            Tuple of (video_velocity, audio_velocity, moe_aux_losses):
                moe_aux_losses is None if MoE is not enabled.
        """
        B = v_latent.shape[0]

        # Timestep embedding
        t_emb = self.t_embed(t)  # (B, dim)

        # Patch embedding
        v_tokens = self.video_patch_embed(v_latent)  # (B, N_v, dim)
        a_tokens = self.audio_patch_embed(a_latent)  # (B, N_a, dim)

        # Compute video grid dimensions
        v_patch_size = self.video_patch_embed.patch_size
        T_v = v_latent.shape[2] // v_patch_size[0]
        H_v = v_latent.shape[3] // v_patch_size[1]
        W_v = v_latent.shape[4] // v_patch_size[2]
        video_grid = (T_v, H_v, W_v)

        # Apply dual-branch transformer layers, collect MoE aux losses
        moe_losses_total = None
        for layer in self.layers:
            result = layer(
                v_tokens, a_tokens, t_emb, text_emb, video_grid,
                mouth_mask=mouth_mask,
            )
            if len(result) == 3:
                v_tokens, a_tokens, moe_aux = result
                if moe_aux is not None:
                    if moe_losses_total is None:
                        moe_losses_total = {}
                    for k, v in moe_aux.items():
                        moe_losses_total[k] = moe_losses_total.get(k, 0.0) + v
            else:
                v_tokens, a_tokens = result

        # Final norm
        v_tokens = self.final_norm_video(v_tokens)
        a_tokens = self.final_norm_audio(a_tokens)

        # Output heads: tokens -> latent patches
        v_velocity_flat = self.video_head(v_tokens)  # (B, N_v, C_v*pT*pH*pW)
        a_velocity_flat = self.audio_head(a_tokens)   # (B, N_a, C_a*pF*pT)

        # Reshape video output back to latent shape
        C_v = v_latent.shape[1]
        pT, pH, pW = v_patch_size
        v_velocity = v_velocity_flat.reshape(B, T_v, H_v, W_v, C_v, pT, pH, pW)
        v_velocity = v_velocity.permute(0, 4, 1, 5, 2, 6, 3, 7)
        v_velocity = v_velocity.reshape(B, C_v, T_v * pT, H_v * pH, W_v * pW)

        # Reshape audio output back to latent shape
        C_a = a_latent.shape[1]
        pF, pTa = self.audio_patch_embed.patch_size
        F_a = a_latent.shape[2] // pF
        T_a_latent = a_latent.shape[3] // pTa
        a_velocity = a_velocity_flat.reshape(B, F_a, T_a_latent, C_a, pF, pTa)
        a_velocity = a_velocity.permute(0, 3, 1, 4, 2, 5)
        a_velocity = a_velocity.reshape(B, C_a, F_a * pF, T_a_latent * pTa)

        # Handle first-frame conditioning for I2VA
        if first_frame_mask is not None:
            # Zero out velocity for the first frame (it's clean)
            v_velocity = v_velocity * first_frame_mask

        return v_velocity, a_velocity, moe_losses_total
