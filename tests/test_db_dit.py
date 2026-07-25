"""Tests for DB-DiT model."""

import torch
import torch.nn as nn
import pytest

from flux.models import DBDiT


class TestDBDiT:
    """Test Dual-Branch Diffusion Transformer forward pass and gradients."""

    @pytest.fixture
    def model(self, device, dtype):
        return DBDiT(
            dim=256,  # Small dim for fast tests
            num_layers=4,
            num_heads=4,
            ffn_ratio=2.0,
            qk_norm=True,
            cbga_layers=[1, 2],
            video_patch_size=(1, 2, 2),
            video_latent_channels=16,
            audio_patch_size=(1, 4),
            audio_latent_channels=8,
        ).to(device).to(dtype)

    def test_forward_shapes(self, model, device, dtype):
        """Test that forward pass produces correct output shapes."""
        B = 2
        # Video latent: (B, 16, T_v=4, H_v=16, W_v=16)
        v_latent = torch.randn(B, 16, 4, 16, 16, device=device, dtype=dtype)
        # Audio latent: (B, 8, F_a=4, T_a=16)
        a_latent = torch.randn(B, 8, 4, 16, device=device, dtype=dtype)
        # Text embeddings: (B, L=16, D=256)
        text_emb = torch.randn(B, 16, 256, device=device, dtype=dtype)
        # Timesteps
        t = torch.rand(B, device=device, dtype=dtype)

        v_vel, a_vel = model(v_latent, a_latent, t, text_emb)

        assert v_vel.shape == v_latent.shape, f"{v_vel.shape} != {v_latent.shape}"
        assert a_vel.shape == a_latent.shape, f"{a_vel.shape} != {a_latent.shape}"

    def test_gradient_flow(self, model, device, dtype):
        """Test that gradients flow through the model."""
        B = 2
        v_latent = torch.randn(B, 16, 4, 16, 16, device=device, dtype=dtype)
        a_latent = torch.randn(B, 8, 4, 16, device=device, dtype=dtype)
        text_emb = torch.randn(B, 16, 256, device=device, dtype=dtype)
        t = torch.rand(B, device=device, dtype=dtype)

        # Set head weights to non-zero for gradient testing
        # (heads are intentionally zero-init for training stability)
        with torch.no_grad():
            nn.init.xavier_uniform_(model.video_head.weight)
            nn.init.xavier_uniform_(model.audio_head.weight)
            model.video_head.bias.data.fill_(0.01)
            model.audio_head.bias.data.fill_(0.01)

        v_vel, a_vel = model(v_latent, a_latent, t, text_emb)
        loss = v_vel.mean() + a_vel.mean()
        loss.backward()

        # Check that gradients flow (at least some non-zero)
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                if torch.any(param.grad != 0):
                    grad_count += 1
        # Expect at least 10% of parameters to have non-zero gradients
        assert grad_count >= 10, f"Only {grad_count} params have non-zero gradients"

    def test_cbga_gates_start_zero(self, model, device, dtype):
        """CBGA gates should be initialized to zero."""
        for i, layer in enumerate(model.layers):
            if layer.cbga is not None:
                assert layer.cbga.v2a_gate.item() == 0.0, f"Layer {i} v2a gate not zero"
                assert layer.cbga.a2v_gate.item() == 0.0, f"Layer {i} a2v gate not zero"

    def test_set_step_updates_gate_scale(self, model, device, dtype):
        """Test CBGA gate warmup scheduling."""
        model.set_step(0)
        for layer in model.layers:
            if layer.cbga is not None:
                assert layer.cbga.get_gate_scale() == 0.0

        model.set_step(50000)
        for layer in model.layers:
            if layer.cbga is not None:
                assert layer.cbga.get_gate_scale() == 1.0

    def test_output_heads_zero_init(self, model, device, dtype):
        """Output heads should be zero-initialized."""
        assert torch.all(model.video_head.weight == 0)
        assert torch.all(model.video_head.bias == 0)
        assert torch.all(model.audio_head.weight == 0)
        assert torch.all(model.audio_head.bias == 0)

    def test_first_frame_mask(self, model, device, dtype):
        """Test I2VA first-frame conditioning mask."""
        B = 2
        v_latent = torch.randn(B, 16, 4, 16, 16, device=device, dtype=dtype)
        a_latent = torch.randn(B, 8, 4, 16, device=device, dtype=dtype)
        text_emb = torch.randn(B, 16, 256, device=device, dtype=dtype)
        t = torch.rand(B, device=device, dtype=dtype)

        # Set head weights to non-zero for mask testing
        with torch.no_grad():
            nn.init.xavier_uniform_(model.video_head.weight)
            nn.init.xavier_uniform_(model.audio_head.weight)
            model.video_head.bias.data.fill_(0.01)
            model.audio_head.bias.data.fill_(0.01)

        # First frame mask: zero out first frame velocity
        mask = torch.ones(1, 1, 4, 1, 1, device=device)
        mask[:, :, 0:1, :, :] = 0.0

        v_vel, a_vel = model(v_latent, a_latent, t, text_emb, first_frame_mask=mask)

        # First frame velocity should be zero
        assert torch.all(v_vel[:, :, 0:1, :, :] == 0)
        # Other frames should have non-zero velocity
        assert not torch.all(v_vel[:, :, 1:, :, :] == 0)
