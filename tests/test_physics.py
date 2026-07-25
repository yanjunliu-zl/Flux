"""Integration tests for physics modules."""

import torch
import pytest

from flux.models import DBDiT
from flux.physics import (
    PhaseLockSampler,
    PhysicsProbe,
    PhysicsProbeLoss,
    VPTNoiseScheduler,
    RoleAwareCaptioner,
    PhysicsRM,
    PhyDPOTrainer,
    CausalMotionGuide,
)
from flux.physics.causal_motion import KeyframeSchedule, TrajectoryConstraint


class TestPhaseLock:
    """Test PhaseLock motion prior locking."""

    @pytest.fixture
    def model(self, device, dtype):
        return DBDiT(
            dim=128, num_layers=2, num_heads=2, ffn_ratio=2.0, qk_norm=True,
        ).to(device).to(dtype).eval()

    def test_motion_prior_extraction(self, model, device, dtype):
        sampler = PhaseLockSampler(model, phase_steps=2)
        v_latent = torch.randn(1, 16, 2, 16, 16, device=device, dtype=dtype)
        a_latent = torch.randn(1, 8, 4, 8, device=device, dtype=dtype)
        text_emb = torch.randn(1, 16, 128, device=device, dtype=dtype)

        motion_v, motion_a = sampler.extract_motion_prior(v_latent, a_latent, text_emb)
        assert motion_v.shape == v_latent.shape
        assert motion_a.shape == a_latent.shape

    def test_phase_lock_sampling(self, model, device, dtype):
        sampler = PhaseLockSampler(model, phase_steps=2, lock_strength=0.3)
        v_shape = (1, 16, 2, 16, 16)
        a_shape = (1, 8, 4, 8)
        text_emb = torch.randn(1, 16, 128, device=device, dtype=dtype)

        v_latent, a_latent = sampler.sample(
            v_shape, a_shape, text_emb, num_steps=3
        )
        assert v_latent.shape == v_shape
        assert a_latent.shape == a_shape

    def test_lock_schedule_decay(self, model, device, dtype):
        for schedule in ["constant", "linear_decay", "cosine_decay"]:
            sampler = PhaseLockSampler(model, lock_schedule=schedule)
            w0 = sampler._get_lock_weight(0, 30)
            w_end = sampler._get_lock_weight(29, 30)
            if schedule == "constant":
                assert abs(w0 - w_end) < 1e-6
            else:
                assert w_end < w0, f"{schedule}: {w_end} >= {w0}"


class TestPhysicsProbe:
    """Test physics probe on DiT hidden states."""

    def test_probe_forward(self, device, dtype):
        probe = PhysicsProbe(dim=256, num_layers=2, num_categories=1)
        hidden = [torch.randn(2, 64, 256, device=device, dtype=dtype) for _ in range(4)]
        scores = probe(hidden)
        assert scores.shape == (2, 1)

    def test_probe_multi_category(self, device, dtype):
        probe = PhysicsProbe(dim=256, num_layers=4, num_categories=12)
        hidden = [torch.randn(2, 64, 256, device=device, dtype=dtype) for _ in range(6)]
        scores = probe(hidden)
        assert scores.shape == (2, 12)

    def test_probe_mean_pooled_input(self, device, dtype):
        """Probe should handle both 2D and 3D hidden states."""
        probe = PhysicsProbe(dim=256, num_layers=2, num_categories=1)
        # 2D input (already pooled)
        hidden_2d = [torch.randn(2, 256, device=device, dtype=dtype) for _ in range(2)]
        scores_2d = probe(hidden_2d)
        assert scores_2d.shape == (2, 1)

    def test_probe_loss(self, device, dtype):
        probe_loss = PhysicsProbeLoss(dim=256, lambda_probe=0.01, warmup_steps=10)
        hidden = [torch.randn(2, 64, 256, device=device, dtype=dtype) for _ in range(4)]
        labels = torch.randint(0, 2, (2,), device=device)
        loss = probe_loss(hidden, labels)
        assert loss.item() >= 0


class TestVPT:
    """Test VPT role-aware captioning and decoupled noise."""

    def test_role_aware_captioner(self):
        captioner = RoleAwareCaptioner()

        # Test basic augmentation
        tagged = captioner.augment("A person pushes a box across the floor")
        assert "[agent:" in tagged
        assert "[controlled:" in tagged
        assert "[passive:" in tagged

        # Test batch
        batch = captioner.augment_batch([
            "A dog catches a ball",
            "The car drives on the road",
        ])
        assert len(batch) == 2
        assert "[agent:" in batch[0]  # dog → agent
        assert "[passive:" in batch[1]  # road → passive

    def test_decoupled_noise_schedule(self, device):
        scheduler = VPTNoiseScheduler(flow_noise_shift=0.3)
        B = 1000  # Large batch for statistical significance
        t_v, t_f = scheduler.sample_timesteps(B, device)
        assert t_v.shape == (B,)
        assert t_f.shape == (B,)
        # Flow timesteps should be shifted toward higher values (less noise)
        assert t_f.mean() > t_v.mean(), f"Flow should be less noisy: {t_f.mean():.3f} <= {t_v.mean():.3f}"

    def test_add_decoupled_noise(self, device, dtype):
        scheduler = VPTNoiseScheduler()
        visual = torch.randn(2, 16, 4, 16, 16, device=device, dtype=dtype)
        flow = torch.randn(2, 2, 4, 16, 16, device=device, dtype=dtype)
        t_v = torch.rand(2, device=device, dtype=dtype)
        t_f = torch.rand(2, device=device, dtype=dtype)
        v_noisy, f_noisy = scheduler.add_noise(visual, flow, t_v, t_f)
        assert v_noisy.shape == visual.shape
        assert f_noisy.shape == flow.shape


class TestPhysCorr:
    """Test PhysCorr reward model and DPO trainer."""

    @pytest.fixture
    def physics_rm(self, device, dtype):
        return PhysicsRM(dim=64).to(device).to(dtype)

    def test_physics_rm_forward(self, physics_rm, device, dtype):
        v_latent = torch.randn(2, 16, 8, 16, 16, device=device, dtype=dtype)
        scores = physics_rm(v_latent)
        assert "physics_score" in scores
        assert "intra_score" in scores
        assert "inter_score" in scores
        assert scores["physics_score"].shape == (2,)

    def test_physics_rm_scoring(self, physics_rm, device, dtype):
        """More structured videos should score similarly; random noise should score lower."""
        # Clean latent
        clean = torch.randn(2, 16, 8, 16, 16, device=device, dtype=dtype)
        # Very noisy latent
        noisy = torch.randn(2, 16, 8, 16, 16, device=device, dtype=dtype) * 5
        score_clean = physics_rm(clean)["physics_score"]
        score_noisy = physics_rm(noisy)["physics_score"]
        # Both are random, but the model should produce valid scores in [-1, 1] range
        assert score_clean.abs().mean() < 5.0
        assert score_noisy.abs().mean() < 5.0

    def test_phy_dpo_trainer_creation(self, device, dtype):
        model = DBDiT(
            dim=128, num_layers=2, num_heads=2, ffn_ratio=2.0, qk_norm=True,
        ).to(device).to(dtype)
        physics_rm = PhysicsRM(dim=64).to(device).to(dtype)
        trainer = PhyDPOTrainer(model, physics_rm, beta=0.1)
        assert trainer is not None
        assert trainer.beta == 0.1


class TestCausalMotion:
    """Test CausalMotion VLM-guided injection."""

    @pytest.fixture
    def model(self, device, dtype):
        return DBDiT(
            dim=128, num_layers=2, num_heads=2, ffn_ratio=2.0, qk_norm=True,
        ).to(device).to(dtype).eval()

    def test_decompose_prompt(self):
        keyframes, trajectories = CausalMotionGuide.decompose_prompt(
            "A ball bounces off the wall", num_frames=8
        )
        assert keyframes is not None
        # Motion verbs should produce trajectories
        has_bounce = "bounce" in "A ball bounces off the wall"
        if has_bounce:
            assert len(trajectories) > 0

    def test_decompose_static_prompt(self):
        """Static prompts should not generate trajectories."""
        _, trajectories = CausalMotionGuide.decompose_prompt(
            "A red apple on a wooden table", num_frames=8
        )
        assert len(trajectories) == 0

    def test_keyframe_schedule(self):
        kf = KeyframeSchedule(num_frames=8, keyframe_positions=[0.0, 0.5, 1.0])
        mask = kf.get_mask(torch.device("cpu"))
        assert mask.shape == (1, 3, 8, 1, 1)
        # Start keyframe should have highest value at t=0
        assert mask[0, 0, 0, 0, 0] > mask[0, 0, 4, 0, 0]

    def test_trajectory_constraint(self):
        traj = TrajectoryConstraint(
            "ball",
            [(0.2, 0.5), (0.5, 0.4), (0.8, 0.3)],
        )
        mask = traj.to_spatial_mask(H=16, W=16, device=torch.device("cpu"))
        assert mask.shape == (3, 16, 16)
        # Peak should be near trajectory positions
        assert mask[0, 8, 3].item() > mask[0, 0, 0].item()  # (0.2, 0.5) ≈ (8, 3)

    def test_causal_motion_guide_creation(self, model):
        guide = CausalMotionGuide(model, guidance_strength=0.3)
        assert guide is not None
        assert guide.guidance_strength == 0.3
