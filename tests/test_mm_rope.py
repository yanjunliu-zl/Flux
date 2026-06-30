"""Tests for MM-RoPE (Multi-Modal Rotary Position Embedding)."""

import torch
import pytest

from seedance.models.db_dit.mm_rope import MMRoPE


class TestMMRoPE:
    """Test MM-RoPE correctness."""

    @pytest.fixture
    def rope(self, device):
        return MMRoPE(
            dim=256,
            rope_dim_t=80,
            rope_dim_h=80,
            rope_dim_w=96,
            rope_dim_a=256,
            theta=10000.0,
        )

    def test_rope_1d_shape(self, rope, device):
        """Test 1D RoPE output shape."""
        x = torch.randn(2, 16, 256, device=device)
        pos = torch.arange(16, device=device, dtype=torch.float32)
        out = rope.apply_rope_1d(x, pos)
        assert out.shape == x.shape

    def test_rope_3d_shape(self, rope, device):
        """Test 3D RoPE output shape."""
        x = torch.randn(2, 4, 8, 8, 256, device=device)
        t_pos = torch.arange(4, device=device, dtype=torch.float32)
        h_pos = torch.arange(8, device=device, dtype=torch.float32)
        w_pos = torch.arange(8, device=device, dtype=torch.float32)
        out = rope.apply_rope_3d(x, t_pos, h_pos, w_pos)
        assert out.shape == x.shape

    def test_rope_preserves_norm(self, rope, device):
        """RoPE should preserve vector norm."""
        x = torch.randn(2, 16, 256, device=device)
        pos = torch.arange(16, device=device, dtype=torch.float32)

        x_norm_before = x.norm(dim=-1)
        out = rope.apply_rope_1d(x, pos)
        x_norm_after = out.norm(dim=-1)

        assert torch.allclose(x_norm_before, x_norm_after, atol=1e-4)

    def test_rope_relative_position(self, rope, device):
        """Test that RoPE encodes relative positions correctly."""
        dim = 256
        # Two tokens at adjacent positions
        x1 = torch.ones(1, dim, device=device)
        x2 = torch.ones(1, dim, device=device)

        pos = torch.tensor([0, 1], device=device, dtype=torch.float32)

        r1 = rope.apply_rope_1d(x1, pos[:1])
        r2 = rope.apply_rope_1d(x2, pos[1:2])

        # They should be different (rotation applied)
        assert not torch.allclose(r1, r2)
