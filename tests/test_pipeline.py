"""Integration tests for the inference pipeline."""

import torch
import pytest

from flux.models import DBDiT, VideoVAE, AudioVAE, T5Encoder


class TestInferencePipeline:
    """Test end-to-end inference pipeline components."""

    @pytest.fixture
    def small_model(self, device, dtype):
        """Create a tiny model for testing."""
        model = DBDiT(
            dim=128,
            num_layers=2,
            num_heads=2,
            ffn_ratio=2.0,
            cbga_layers=[0],
            video_patch_size=(1, 2, 2),
            video_latent_channels=16,
            audio_patch_size=(1, 4),
            audio_latent_channels=8,
        ).to(device).to(dtype)
        return model

    @pytest.fixture
    def small_vae_video(self, device, dtype):
        return VideoVAE(
            in_channels=3,
            latent_channels=16,
            base_channels=16,
            channel_multipliers=[1, 2],
            spatial_strides=[2, 2],
            temporal_strides=[2, 1],
            num_res_blocks=1,
        ).to(device).to(dtype)

    @pytest.fixture
    def small_vae_audio(self, device, dtype):
        return AudioVAE(
            latent_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2],
            strides=[[1, 2], [2, 2]],
            num_res_blocks=1,
        ).to(device).to(dtype)

    def test_t2va_pipeline_creation(
        self, small_model, small_vae_video, small_vae_audio, device, dtype
    ):
        """Test that T2VA pipeline can be created without error."""
        from flux.pipelines import T2VAPipeline

        # Skip T5 loading (requires network) — use None for text encoder
        class DummyTextEncoder:
            def __call__(self, texts):
                return torch.randn(len(texts), 16, 128)

            def to(self, device):
                return self

            def eval(self):
                return self

        pipeline = T2VAPipeline(
            vae_video=small_vae_video,
            vae_audio=small_vae_audio,
            db_dit=small_model,
            text_encoder=DummyTextEncoder(),
            device=device,
            dtype=dtype,
        )
        assert pipeline is not None

    def test_model_forward_inference_mode(
        self, small_model, device, dtype
    ):
        """Test model forward pass in inference mode."""
        small_model.eval()
        B = 1
        v_latent = torch.randn(B, 16, 2, 16, 16, device=device, dtype=dtype)
        a_latent = torch.randn(B, 8, 2, 8, device=device, dtype=dtype)
        text_emb = torch.randn(B, 16, 128, device=device, dtype=dtype)
        t = torch.zeros(B, device=device, dtype=dtype)

        with torch.no_grad():
            v_vel, a_vel = small_model(v_latent, a_latent, t, text_emb)

        assert v_vel.shape == v_latent.shape
        assert a_vel.shape == a_latent.shape
