"""Text-to-Video-Audio (T2VA) inference pipeline.

Orchestrates the full generation process:
Text → T5 encode → DB-DiT sampling (CFG) → VideoVAE decode + AudioVAE decode → MP4 output
"""

import torch
import torch.nn as nn

from seedance.diffusion.flow_matching import FlowMatching
from seedance.utils.video_utils import save_video
from seedance.utils.audio_utils import save_audio


class T2VAPipeline:
    """End-to-end text-to-video-audio generation pipeline.

    Args:
        vae_video: VideoVAE model.
        vae_audio: AudioVAE model.
        db_dit: DB-DiT model.
        text_encoder: T5 text encoder.
        device: Target device.
        dtype: Model dtype.
    """

    def __init__(
        self,
        vae_video: nn.Module,
        vae_audio: nn.Module,
        db_dit: nn.Module,
        text_encoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.vae_video = vae_video.to(device=device, dtype=dtype).eval()
        self.vae_audio = vae_audio.to(device=device, dtype=dtype).eval()
        self.db_dit = db_dit.to(device=device, dtype=dtype).eval()
        self.text_encoder = text_encoder.to(device=device).eval()
        self.device = device
        self.dtype = dtype

        self.flow_matching = FlowMatching()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 32,
        width: int = 256,
        height: int = 256,
        fps: int = 16,
        num_steps: int = 30,
        sampler: str = "heun",
        cfg_video: float = 5.0,
        cfg_audio: float = 4.0,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate video and audio from text prompt.

        Args:
            prompt: Text description.
            negative_prompt: Negative prompt for CFG.
            num_frames: Number of video frames.
            width: Video width.
            height: Video height.
            fps: Frames per second.
            num_steps: Diffusion sampling steps.
            sampler: "euler" or "heun".
            cfg_video: CFG scale for video.
            cfg_audio: CFG scale for audio.
            seed: Random seed.

        Returns:
            Tuple of (video_frames, audio_waveform).
        """
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Encode text
        pos_emb = self.text_encoder([prompt]).to(dtype=self.dtype)
        neg_emb = self.text_encoder([negative_prompt or ""]).to(dtype=self.dtype)

        # Latent shapes
        v_shape = (
            1,
            self.vae_video.latent_channels,
            max(1, num_frames // 4),
            max(8, height // 8),
            max(8, width // 8),
        )
        # Audio: compute shape compatible with patch_size (1, 4)
        # AudioVAE has ~8x temporal compression from mel frames
        audio_duration_s = num_frames / fps
        audio_samples = int(audio_duration_s * self.vae_audio.sample_rate)
        a_frames = self.vae_audio.mel_transform.get_output_length(audio_samples)
        # T_a must be divisible by audio patch time stride (4)
        t_a = max(1, a_frames // 8)
        t_a = ((t_a + 3) // 4) * 4  # round up to multiple of 4
        a_shape = (
            1,
            self.vae_audio.latent_channels,
            4,  # F_a — must be divisible by freq patch (1), keep it simple
            t_a,
        )

        # Sample via flow matching ODE
        v_latent, a_latent = self.flow_matching.sample(
            model=self.db_dit,
            v_shape=v_shape,
            a_shape=a_shape,
            text_emb=pos_emb,
            null_text_emb=neg_emb,
            num_steps=num_steps,
            cfg_video=cfg_video,
            cfg_audio=cfg_audio,
            sampler=sampler,
        )

        # Decode video only (audio branch not trained in Stage 1)
        video_frames = self.vae_video.decode(v_latent)
        audio_waveform = None

        return video_frames, audio_waveform

    def generate_to_file(
        self,
        prompt: str,
        output_path: str,
        **kwargs,
    ):
        """Generate and save video to MP4 file.

        Args:
            prompt: Text prompt.
            output_path: Output MP4 file path.
            **kwargs: Passed to generate().
        """
        video_frames, _ = self.generate(prompt, **kwargs)

        # Save video
        video_frames = video_frames[0]  # Remove batch dim
        if video_frames.shape[0] == 3:  # (C, T, H, W) -> (T, C, H, W)
            video_frames = video_frames.permute(1, 0, 2, 3)

        save_video(video_frames, output_path, fps=kwargs.get("fps", 16))
        print(f"Generated: {output_path}")
