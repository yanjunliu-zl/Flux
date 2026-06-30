"""Tests for VideoVAE."""

import torch
import pytest

from seedance.models import VideoVAE


class TestVideoVAE:
    """Test VideoVAE encode/decode and forward pass."""

    @pytest.fixture
    def vae(self, device, dtype):
        return VideoVAE(
            in_channels=3,
            latent_channels=16,
            base_channels=32,  # Small for tests
            channel_multipliers=[1, 2, 2],
            spatial_strides=[1, 2, 2],
            temporal_strides=[2, 1, 1],
            num_res_blocks=1,
        ).to(device).to(dtype)

    def test_encode_shape(self, vae, device, dtype):
        """Test encoder output shapes."""
        x = torch.randn(2, 3, 16, 64, 64, device=device, dtype=dtype)
        mean, logvar = vae.encoder(x)
        # strides [2,1,1] → 2x temporal, strides [1,2,2] → 4x spatial
        assert mean.shape == (2, 16, 8, 16, 16), f"Got {mean.shape}"
        assert logvar.shape == (2, 16, 8, 16, 16), f"Got {logvar.shape}"

    def test_decode_shape(self, vae, device, dtype):
        """Test decoder output shapes."""
        z = torch.randn(2, 16, 8, 16, 16, device=device, dtype=dtype)
        out = vae.decode(z)
        assert out.shape == (2, 3, 16, 64, 64)

    def test_forward_roundtrip(self, vae, device, dtype):
        """Test full encode-decode roundtrip."""
        x = torch.randn(2, 3, 16, 64, 64, device=device, dtype=dtype)
        recon, z, kl = vae(x, sample=True)
        assert recon.shape == x.shape
        assert z.shape == (2, 16, 8, 16, 16)
        assert kl.ndim == 0  # Scalar KL loss

    def test_encode_deterministic(self, vae, device, dtype):
        """Test that encode(sample=False) is deterministic."""
        x = torch.randn(2, 3, 16, 64, 64, device=device, dtype=dtype)
        z1 = vae.encode(x, sample=False)
        z2 = vae.encode(x, sample=False)
        assert torch.allclose(z1, z2)

    def test_kl_loss_positive(self, vae, device, dtype):
        """KL divergence should be non-negative."""
        x = torch.randn(2, 3, 16, 64, 64, device=device, dtype=dtype)
        _, _, kl = vae(x)
        assert kl.item() >= 0

    def test_gradient_flow(self, vae, device, dtype):
        """Verify gradients flow through the VAE."""
        x = torch.randn(2, 3, 16, 64, 64, device=device, dtype=dtype)
        recon, _, _ = vae(x)
        loss = ((recon - x) ** 2).mean()
        loss.backward()

        # Check grads exist
        for name, param in vae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
