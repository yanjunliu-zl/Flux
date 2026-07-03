"""Tests for Flow Matching framework."""

import torch
import pytest

from seedance.diffusion.flow_matching import FlowMatching
from seedance.diffusion.noise_schedule import LogitNormalSchedule
from seedance.models import DBDiT


class TestFlowMatching:
    """Test flow matching loss and sampling."""

    @pytest.fixture
    def fm(self):
        return FlowMatching(
            video_weight=1.0,
            audio_weight=1.0,
            sync_weight=0.0,
        )

    @pytest.fixture
    def model(self, device, dtype):
        return DBDiT(
            dim=256, num_layers=4, num_heads=4, ffn_ratio=2.0,
            qk_norm=True,
            video_patch_size=(1, 2, 2), video_latent_channels=16,
            audio_patch_size=(1, 4), audio_latent_channels=8,
        ).to(device).to(dtype)

    def test_loss_decreases(self, fm, model, device, dtype):
        """Flow matching loss should decrease with optimization."""
        B = 2
        # Fix seed for deterministic test
        torch.manual_seed(42)
        v_clean = torch.randn(B, 16, 4, 16, 16, device=device, dtype=dtype)
        a_clean = torch.randn(B, 8, 4, 16, device=device, dtype=dtype)
        text_emb = torch.randn(B, 16, 256, device=device, dtype=dtype)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Initial loss
        losses_before = fm.get_training_loss(model, v_clean, a_clean, text_emb)
        loss_before = losses_before["loss"].item()

        # Optimize enough steps for convergence on the fixed sample
        for _ in range(10):
            losses = fm.get_training_loss(model, v_clean, a_clean, text_emb)
            loss = losses["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss after training
        losses_after = fm.get_training_loss(model, v_clean, a_clean, text_emb)
        loss_after = losses_after["loss"].item()

        assert loss_after < loss_before, f"Loss should decrease: {loss_after} >= {loss_before}"

    def test_sample_output_shape(self, fm, model, device, dtype):
        """Test that sampling produces correct output shapes."""
        v_shape = (1, 16, 2, 16, 16)
        a_shape = (1, 8, 4, 8)
        text_emb = torch.randn(1, 16, 256, device=device, dtype=dtype)

        v_latent, a_latent = fm.sample(
            model=model,
            v_shape=v_shape,
            a_shape=a_shape,
            text_emb=text_emb,
            num_steps=4,  # Small for test speed
            cfg_video=1.0,
            cfg_audio=1.0,
            sampler="euler",
        )

        assert v_latent.shape == v_shape
        assert a_latent.shape == a_shape

    def test_logit_normal_schedule(self, device):
        """Test logit-normal timestep sampling."""
        schedule = LogitNormalSchedule()
        samples = schedule.sample(10000, device)

        # Should be in [0, 1]
        assert samples.min() >= 0
        assert samples.max() <= 1

        # Should concentrate around 0.5
        assert 0.3 < samples.mean() < 0.7
