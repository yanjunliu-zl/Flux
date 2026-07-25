"""Noise schedule for flow matching.

Uses logit-normal timestep sampling, which concentrates training
on the intermediate noise levels where the task is hardest.
"""

import torch
import torch.nn as nn


class LogitNormalSchedule(nn.Module):
    """Logit-normal timestep sampling for flow matching.

    Samples t = sigmoid(eps) where eps ~ N(loc, scale).
    This biases sampling toward t=0.5 (intermediate noise levels).

    Args:
        loc: Mean of the underlying normal distribution (default: 0.0).
        scale: Standard deviation of the underlying normal (default: 1.0).
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample.
            device: Target device.

        Returns:
            Timesteps tensor of shape (batch_size,), values in [0, 1].
        """
        eps = self.loc + self.scale * torch.randn(batch_size, device=device)
        t = torch.sigmoid(eps)
        return t

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """Sample timesteps with same batch size as input.

        Args:
            x: Input tensor, uses batch size from first dimension.

        Returns:
            Timesteps (batch_size,).
        """
        return self.sample(x.shape[0], x.device)
