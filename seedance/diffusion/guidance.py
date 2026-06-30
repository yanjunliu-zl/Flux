"""Classifier-Free Guidance (CFG) utilities.

During training, randomly drops text conditioning to enable CFG at inference.
During inference, interpolates between conditional and unconditional predictions.
"""

import torch


def classifier_free_guidance(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    a_cond: torch.Tensor,
    a_uncond: torch.Tensor,
    cfg_video: float = 5.0,
    cfg_audio: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply classifier-free guidance to video and audio velocity predictions.

    v_cfg = v_uncond + cfg_video * (v_cond - v_uncond)
    a_cfg = a_uncond + cfg_audio * (a_cond - a_uncond)

    Args:
        v_cond: Conditional video velocity prediction.
        v_uncond: Unconditional video velocity prediction.
        a_cond: Conditional audio velocity prediction.
        a_uncond: Unconditional audio velocity prediction.
        cfg_video: CFG scale for video (>1 increases prompt adherence).
        cfg_audio: CFG scale for audio.

    Returns:
        Tuple of (guided_video_velocity, guided_audio_velocity).
    """
    v_cfg = v_uncond + cfg_video * (v_cond - v_uncond)
    a_cfg = a_uncond + cfg_audio * (a_cond - a_uncond)
    return v_cfg, a_cfg
