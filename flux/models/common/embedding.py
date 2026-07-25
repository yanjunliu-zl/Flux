"""Patch embedding and timestep embedding modules."""

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Convert image/video/audio latent to patch tokens via convolution.

    Args:
        in_channels: Number of input channels.
        embed_dim: Output embedding dimension.
        patch_size: Patch size (T, H, W) for video or (F, T) for audio.
        spatial_only: If True, 2D patch embed (no temporal). For audio, always False.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, ...],
        spatial_only: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_only = spatial_only

        if spatial_only or len(patch_size) == 2:
            # 2D patch embedding
            self.proj = nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size,
            )
        else:
            # 3D patch embedding
            self.proj = nn.Conv3d(
                in_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) for video, (B, C, F, T_a) for audio
        x = self.proj(x)  # (B, D, T', H', W') or (B, D, F', T_a')
        # Flatten spatial/temporal dims
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP.

    Embeds a scalar timestep t into a vector of dimension `dim`.
    Uses sinusoidal encoding (like in Transformers) followed by a 2-layer MLP.

    Args:
        dim: Output embedding dimension.
        frequency_embedding_size: Intermediate frequency dimension (default: 256).
        max_period: Maximum period for sinusoidal encoding (default: 10000).
    """

    def __init__(
        self,
        dim: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timestep tensor of shape (B,) or (B, 1), values in [0, 1].
            dim: Embedding dimension.
            max_period: Maximum period.

        Returns:
            Embedding tensor of shape (B, dim).
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)

        args = t.float()[:, None] * freqs[None, :]  # (B, 1) * (1, half) -> (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(t_emb.to(self.mlp[0].weight.dtype))
