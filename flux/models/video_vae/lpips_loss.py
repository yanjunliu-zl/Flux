"""LPIPS perceptual loss wrapper for VAE training.

LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual
similarity between images using deep network features.

For video, we compute LPIPS frame-by-frame and average.
"""

import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss for video.

    Computes LPIPS per-frame and averages across the temporal dimension.

    Args:
        net: LPIPS network type ('alex', 'vgg', 'squeeze'). Default: 'alex'.
        spatial: Whether to return spatial map or scalar. Default: False (scalar).
    """

    def __init__(self, net: str = "alex", spatial: bool = False):
        super().__init__()
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net, spatial=spatial)
        except ImportError:
            raise ImportError(
                "lpips package is required. Install with: pip install lpips"
            )
        # Freeze LPIPS network
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss between predicted and target videos.

        Args:
            pred: Predicted video (B, 3, T, H, W), values in [0, 1] or [-1, 1].
            target: Target video (B, 3, T, H, W), same range.

        Returns:
            Scalar LPIPS loss averaged over frames and batch.
        """
        B, C, T, H, W = pred.shape

        # Normalize to [-1, 1] if needed (LPIPS expects this range)
        if pred.min() >= 0 and pred.max() <= 1:
            pred = 2 * pred - 1
            target = 2 * target - 1

        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        loss = self.lpips_fn(pred_2d, target_2d)
        return loss.mean()
