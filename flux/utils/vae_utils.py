"""VideoVAE utilities for building, loading pretrained weights, and encoding/decoding.

Provides:
- build_vae_from_config: Build VideoVAE from YAML config or defaults.
- load_sdxl_vae_weights: Load SDXL 2D VAE weights from HuggingFace diffusers.
- encode_video: Encode video tensor to latent (with fallback).
- decode_latent: Decode latent tensor to video frames (with fallback).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_video_vae(
    *,
    latent_channels: int = 16,
    base_channels: int = 128,
    channel_multipliers: list[int] | None = None,
    spatial_strides: list[int] | None = None,
    temporal_strides: list[int] | None = None,
    num_res_blocks: int = 2,
    attn_resolutions: list[int] | None = None,
    norm_groups: int = 32,
    kl_weight: float = 1e-6,
    pretrained_path: str | None = None,
    sdxl_vae_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Build a VideoVAE model, optionally loading pretrained weights.

    Args:
        latent_channels: Latent space channels (default 16, matches SDXL's 4).
        base_channels: Base convolution channels.
        channel_multipliers: Per-stage channel multipliers.
        spatial_strides: Per-stage spatial strides (cumulative 8x spatial compression).
        temporal_strides: Per-stage temporal strides (cumulative temporal compression).
            Use [1, 1, 1, 1] for per-frame encoding (no temporal compression).
            Use [2, 2, 1, 1] for 4x temporal compression.
        num_res_blocks: Number of ResBlocks per stage.
        attn_resolutions: Resolutions at which to apply spatial attention.
        norm_groups: GroupNorm groups.
        kl_weight: KL divergence weight.
        pretrained_path: Path to a VideoVAE checkpoint (.pt file).
        sdxl_vae_model: HuggingFace model ID for SDXL VAE (used only if no pretrained_path).
        device: Target device.
        dtype: Model dtype (default: float32 for VAE).

    Returns:
        VideoVAE model in eval mode, frozen (no grad).
    """
    from flux.models.video_vae import VideoVAE

    if channel_multipliers is None:
        channel_multipliers = [1, 2, 4, 4]
    if spatial_strides is None:
        spatial_strides = [1, 2, 2, 2]
    if temporal_strides is None:
        temporal_strides = [1, 1, 1, 1]  # Default: no temporal compression
    if attn_resolutions is None:
        attn_resolutions = [16]

    vae = VideoVAE(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        spatial_strides=spatial_strides,
        temporal_strides=temporal_strides,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        norm_groups=norm_groups,
        kl_weight=kl_weight,
    )

    if pretrained_path is not None:
        # Load from a VideoVAE checkpoint
        print(f"[VideoVAE] Loading pretrained weights from {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        # Filter to VAE keys if checkpoint contains other models too
        vae_state = {k: v for k, v in state.items() if not k.startswith("db_dit")}
        if vae_state:
            missing, unexpected = vae.load_state_dict(vae_state, strict=False)
            if missing:
                print(f"  [VideoVAE] Missing keys: {len(missing)}")
            if unexpected:
                print(f"  [VideoVAE] Unexpected keys: {len(unexpected)}")
    else:
        # Try to initialize from SDXL 2D VAE
        try:
            sdxl_state = _load_sdxl_vae_weights(sdxl_vae_model)
            if sdxl_state is not None:
                vae.init_from_sdxl_vae(sdxl_state)
        except Exception as e:
            print(f"  [VideoVAE] Could not load SDXL VAE ({e}), using random init")

    # Freeze VAE — it's used for encoding/decoding only, no gradients
    for param in vae.parameters():
        param.requires_grad = False

    if device is not None:
        vae = vae.to(device=device)
    if dtype is not None:
        vae = vae.to(dtype=dtype)

    vae.eval()

    # Compute and print compression stats
    spatial_comp = 1
    for s in spatial_strides:
        spatial_comp *= s
    temporal_comp = 1
    for s in temporal_strides:
        temporal_comp *= s
    total_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(
        f"[VideoVAE] Built: {total_params:.1f}M params, "
        f"{spatial_comp}x spatial, {temporal_comp}x temporal compression, "
        f"{latent_channels} latent channels"
    )

    return vae


def _vae_dtype(vae: nn.Module) -> torch.dtype:
    """Get the dtype of the VAE's first parameter."""
    return next(vae.parameters()).dtype


def encode_video(
    vae: nn.Module | None,
    video: torch.Tensor,
    use_mean: bool = True,
) -> torch.Tensor:
    """Encode video to latent using VAE, with bilinear fallback.

    The VAE must produce latents with the same number of channels as the
    downstream model expects. For SDXL-initialized VAEs this is 16.

    Args:
        vae: VideoVAE model or None for bilinear fallback.
        video: Video tensor (B, C, T, H, W) in [-1, 1] range.
        use_mean: If True, use posterior mean (deterministic). If False, sample.

    Returns:
        Latent tensor (B, latent_channels, T_latent, H_latent, W_latent).
    """
    if vae is not None:
        with torch.no_grad():
            # Match VAE dtype for the forward pass
            vae_dtype = _vae_dtype(vae)
            video_in = video.to(dtype=vae_dtype)
            latent = vae.encode(video_in, sample=not use_mean)
            # Return latent in the original input dtype for downstream model
            return latent.to(dtype=video.dtype)
    else:
        # Fallback: bilinear downscale + zero-pad (legacy behavior)
        return _bilinear_encode(video)


def decode_latent(
    vae: nn.Module | None,
    latent: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Decode latent to video frames using VAE, with bilinear fallback.

    Args:
        vae: VideoVAE model or None for bilinear fallback.
        latent: Latent tensor (B, C, T_latent, H_latent, W_latent).
        target_height: Target frame height.
        target_width: Target frame width.

    Returns:
        Video tensor (B, C, T, target_height, target_width) in [-1, 1] range.
    """
    if vae is not None and not _is_fake_vae(vae):
        with torch.no_grad():
            vae_dtype = _vae_dtype(vae)
            latent_in = latent.to(dtype=vae_dtype)
            output = vae.decode(latent_in)
            return output.to(dtype=latent.dtype)
    else:
        return _bilinear_decode(latent, target_height, target_width)


def _bilinear_encode(video: torch.Tensor) -> torch.Tensor:
    """Legacy bilinear downscale + zero-pad encoding.

    video: (B, C, T, H, W) -> latent: (B, 16, T, H//8, W//8)
    """
    B, C, T, H, W = video.shape
    v_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    v_flat = F.interpolate(
        v_flat, size=(H // 8, W // 8), mode='bilinear', antialias=True,
    )
    v_latent = v_flat.reshape(B, T, C, H // 8, W // 8).permute(0, 2, 1, 3, 4)
    v_latent = torch.cat([
        v_latent,
        torch.zeros(B, 13, T, H // 8, W // 8, device=video.device, dtype=video.dtype),
    ], dim=1)
    return v_latent


def _bilinear_decode(
    latent: torch.Tensor, target_height: int, target_width: int,
) -> torch.Tensor:
    """Legacy bilinear upscale from first 3 channels.

    latent: (B, C, T, H', W') -> video: (B, 3, T, target_h, target_w)
    """
    latent = latent[:, :3]  # Take first 3 channels (RGB)
    B, C, T, H_small, W_small = latent.shape
    v_flat = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H_small, W_small)
    video_frames = F.interpolate(
        v_flat, size=(target_height, target_width), mode='bilinear', antialias=True,
    )
    video_frames = video_frames.reshape(B, T, C, target_height, target_width)
    video_frames = video_frames.permute(0, 2, 1, 3, 4)
    return video_frames


def _is_fake_vae(vae: nn.Module) -> bool:
    """Check if the VAE is a FakeVAE stub."""
    return type(vae).__name__ == "FakeVAE"


def _load_sdxl_vae_weights(model_id: str = "stabilityai/sdxl-vae") -> dict | None:
    """Load SDXL VAE weights from HuggingFace diffusers.

    Args:
        model_id: HF model ID for the SDXL VAE.

    Returns:
        State dict of the SDXL VAE, or None if loading fails.
    """
    try:
        from diffusers import AutoencoderKL
        # Try with subfolder="vae" first (for full SDXL pipeline checkpoints),
        # then without (for standalone VAE checkpoints like stabilityai/sdxl-vae)
        try:
            vae_2d = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        except Exception:
            vae_2d = AutoencoderKL.from_pretrained(model_id)
        state = vae_2d.state_dict()
        print(f"  [VideoVAE] Loaded SDXL VAE weights from {model_id}")
        return state
    except ImportError:
        print("  [VideoVAE] diffusers not installed, skipping SDXL VAE init")
        return None
    except Exception as e:
        print(f"  [VideoVAE] Failed to load SDXL VAE: {e}")
        return None
