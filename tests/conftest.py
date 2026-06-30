"""Shared test fixtures and configuration."""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return torch.float32


@pytest.fixture
def batch_size():
    """Small batch size for tests."""
    return 2


@pytest.fixture
def video_shape():
    """Default video input shape: (B, C, T, H, W)."""
    return (2, 3, 16, 256, 256)


@pytest.fixture
def audio_shape():
    """Default mel-spectrogram shape: (B, 1, n_mels, T_frames)."""
    return (2, 1, 80, 256)


@pytest.fixture
def video_latent_shape():
    """Video latent shape after VAE encoding."""
    return (2, 16, 4, 32, 32)  # T/4, H/8, W/8


@pytest.fixture
def audio_latent_shape():
    """Audio latent shape after VAE encoding."""
    return (2, 8, 5, 32)
