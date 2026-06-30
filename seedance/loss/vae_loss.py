"""VAE loss functions combining reconstruction, perceptual, KL, and GAN terms.

Used for both VideoVAE and AudioVAE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """Combined VAE loss with L1, LPIPS, KL, and GAN components.

    Args:
        l1_weight: Weight for L1 reconstruction loss.
        lpips_weight: Weight for LPIPS perceptual loss.
        kl_weight: Weight for KL divergence regularization.
        gan_weight: Weight for GAN discriminator loss.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 1.0,
        kl_weight: float = 1e-6,
        gan_weight: float = 0.5,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.kl_weight = kl_weight
        self.gan_weight = gan_weight

    def reconstruction_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """L1 reconstruction loss."""
        return F.l1_loss(pred, target)

    def perceptual_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """LPIPS perceptual loss (requires lpips package)."""
        try:
            import lpips
            if not hasattr(self, "lpips_fn"):
                self.lpips_fn = lpips.LPIPS(net="alex", spatial=False).to(pred.device)
            # Handle 5D video: reshape to (B*T, C, H, W)
            if pred.dim() == 5:
                B, C, T, H, W = pred.shape
                pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                return self.lpips_fn(pred_2d, target_2d).mean()
            return self.lpips_fn(pred, target).mean()
        except ImportError:
            return torch.tensor(0.0, device=pred.device)

    def gan_generator_loss(
        self, fake_preds: list[torch.Tensor]
    ) -> torch.Tensor:
        """Hinge loss for generator: -E[D(G(z))]."""
        # Use the last layer output (logits)
        return -fake_preds[-1].mean()

    def gan_discriminator_loss(
        self,
        real_preds: list[torch.Tensor],
        fake_preds: list[torch.Tensor],
    ) -> torch.Tensor:
        """Hinge loss for discriminator."""
        real_loss = F.relu(1.0 - real_preds[-1]).mean()
        fake_loss = F.relu(1.0 + fake_preds[-1]).mean()
        return (real_loss + fake_loss) / 2

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        kl: torch.Tensor,
        fake_preds: list[torch.Tensor] | None = None,
        real_preds: list[torch.Tensor] | None = None,
        training_generator: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute full VAE loss.

        Args:
            recon: Reconstructed output.
            target: Ground truth.
            kl: KL divergence (scalar or per-sample).
            fake_preds: Discriminator outputs for generated samples.
            real_preds: Discriminator outputs for real samples.
            training_generator: If True, compute generator loss; else discriminator.

        Returns:
            Dict of loss components.
        """
        l1_loss = self.reconstruction_loss(recon, target)
        perc_loss = self.perceptual_loss(recon, target)

        loss = self.l1_weight * l1_loss + self.lpips_weight * perc_loss

        if kl is not None:
            loss = loss + self.kl_weight * kl

        gan_loss = torch.tensor(0.0, device=recon.device)
        if fake_preds is not None and self.gan_weight > 0:
            if training_generator:
                gan_loss = self.gan_generator_loss(fake_preds)
                loss = loss + self.gan_weight * gan_loss
            elif real_preds is not None:
                gan_loss = self.gan_discriminator_loss(real_preds, fake_preds)
                loss = gan_loss

        return {
            "loss": loss,
            "l1_loss": l1_loss.detach(),
            "perceptual_loss": perc_loss.detach(),
            "kl_loss": kl.detach() if isinstance(kl, torch.Tensor) else kl,
            "gan_loss": gan_loss.detach() if isinstance(gan_loss, torch.Tensor) else gan_loss,
        }
