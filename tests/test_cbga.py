"""Tests for Cross-Branch Gated Attention (CBGA)."""

import torch
import pytest

from seedance.models.db_dit.cross_modal_bridge import CBGABlock


class TestCBGA:
    """Test CBGA module."""

    @pytest.fixture
    def cbga(self, device):
        return CBGABlock(
            dim=256,
            num_heads=4,
            qk_norm=True,
        ).to(device)

    def test_gates_initialized_zero(self, cbga):
        """CBGA gates should be zero at initialization."""
        assert cbga.v2a_gate.item() == 0.0
        assert cbga.a2v_gate.item() == 0.0

    def test_zero_gate_no_effect(self, cbga, device):
        """With zero gates, output should equal input (residual pass-through)."""
        B, N_v, N_a, D = 2, 64, 32, 256
        v_tokens = torch.randn(B, N_v, D, device=device)
        a_tokens = torch.randn(B, N_a, D, device=device)
        t_emb = torch.randn(B, D, device=device)

        v_out, a_out = cbga(v_tokens, a_tokens, t_emb)

        # With zero gates (and zero warmup), outputs should equal inputs
        assert torch.allclose(v_out, v_tokens)
        assert torch.allclose(a_out, a_tokens)

    def test_gate_warmup(self, cbga, device):
        """Gate scale should follow warmup schedule."""
        cbga.set_step(0)
        assert cbga.get_gate_scale() == 0.0

        cbga.set_step(25000)
        assert cbga.get_gate_scale() == 0.5

        cbga.set_step(50000)
        assert cbga.get_gate_scale() == 1.0

    def test_shape_preserved(self, cbga, device):
        """CBGA should preserve input shapes."""
        B, N_v, N_a, D = 2, 64, 32, 256
        v_tokens = torch.randn(B, N_v, D, device=device)
        a_tokens = torch.randn(B, N_a, D, device=device)
        t_emb = torch.randn(B, D, device=device)

        v_out, a_out = cbga(v_tokens, a_tokens, t_emb)
        assert v_out.shape == v_tokens.shape
        assert a_out.shape == a_tokens.shape
