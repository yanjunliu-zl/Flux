"""Image-to-Video-Audio (I2VA) inference pipeline.

Extends T2VA with first-frame conditioning from an input image.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from seedance.pipelines.pipeline_t2va import T2VAPipeline
from seedance.utils.video_utils import save_video
from seedance.utils.audio_utils import save_audio


class I2VAPipeline(T2VAPipeline):
    """Image-to-video-audio generation pipeline.

    Conditions generation on an input image as the first frame.
    Inherits from T2VAPipeline and adds image preprocessing.
    """

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor | str | Image.Image,
        prompt: str = "",
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
        """Generate video from image and text.

        Args:
            image: Input image (tensor: C,H,W or 1,C,H,W), file path, or PIL Image.
            prompt: Text describing desired motion.
            negative_prompt: Negative prompt for CFG.
            num_frames: Number of output video frames.
            width: Output width.
            height: Output height.
            fps: Frames per second.
            num_steps: Diffusion steps.
            sampler: "euler" or "heun".
            cfg_video: CFG scale for video.
            cfg_audio: CFG scale for audio.
            seed: Random seed.

        Returns:
            Tuple of (video_frames, audio_waveform).
        """
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Resize to target resolution
        image = TF.resize(image, [height, width], antialias=True)

        # Normalize to [-1, 1]
        image = 2.0 * image - 1.0
        image = image.to(self.device, dtype=self.dtype)
        image = image.unsqueeze(2)  # (B, C, 1, H, W)

        # Encode image to latent via VideoVAE (treat as single-frame video)
        first_frame_latent = self.vae_video.encode(image, sample=False)
        first_frame_latent = first_frame_latent.squeeze(2)  # Remove temporal dim

        # Set seed
        torch.manual_seed(seed)

        # Encode text
        pos_emb = self.text_encoder([prompt]).to(dtype=self.dtype)
        neg_emb = self.text_encoder([negative_prompt or ""]).to(dtype=self.dtype)

        # Latent shapes
        v_shape = (
            1,
            self.vae_video.latent_channels,
            num_frames // 4,
            height // 8,
            width // 8,
        )
        audio_duration_s = num_frames / fps
        audio_samples = int(audio_duration_s * self.vae_audio.sample_rate)
        a_frames = self.vae_audio.mel_transform.get_output_length(audio_samples)
        a_shape = (
            1,
            self.vae_audio.latent_channels,
            5,
            a_frames // 8,
        )

        # Sample with first-frame conditioning
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
            first_frame_latent=first_frame_latent,
        )

        # Decode
        video_frames = self.vae_video.decode(v_latent.to(self.dtype))
        audio_waveform = self.vae_audio.latent_to_waveform(a_latent.to(self.dtype))

        return video_frames, audio_waveform
