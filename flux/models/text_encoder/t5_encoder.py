"""T5 Text Encoder wrapper for text conditioning.

Uses HuggingFace Transformers to load a T5 model (frozen during training).
Outputs the last hidden state for cross-attention in DB-DiT.
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class T5Encoder(nn.Module):
    """T5 text encoder for conditioning the diffusion model.

    Loads a pretrained T5 model (e.g., google/t5-v1_1-xxl) and freezes its weights.
    Returns last_hidden_state for cross-attention conditioning.

    Args:
        model_name: HuggingFace T5 model ID (default: "google/t5-v1_1-xxl").
        max_length: Maximum token length (default: 226).
        dim: Output projection dimension. If not None, project T5 output to this dim.
        device: Device to load model on ("cpu", "cuda").
        dtype: Model dtype (torch.float16, torch.bfloat16, etc.).
    """

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-xxl",
        max_length: int = 226,
        dim: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Load encoder-only T5
        self.encoder = T5EncoderModel.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.hidden_size = self.encoder.config.d_model

        # Freeze T5 parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Optional projection to model dimension
        if dim is not None and dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, dim)
        else:
            self.proj = nn.Identity()

        self.dim = dim or self.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text prompts.

        Args:
            texts: List of text strings.

        Returns:
            Text embeddings (B, L, hidden_size) — last_hidden_state.
        """
        tokens = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.encoder.device)

        outputs = self.encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        # Last hidden state: (B, seq_len, hidden_size)
        return outputs.last_hidden_state

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode and project text.

        Args:
            texts: List of text strings.

        Returns:
            Text embeddings (B, L, dim).
        """
        hidden = self.encode(texts)
        return self.proj(hidden)

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get null text embedding for classifier-free guidance.

        Args:
            batch_size: Batch size.
            device: Target device.

        Returns:
            Zero embedding of shape (B, max_length, dim).
        """
        return torch.zeros(
            batch_size, self.max_length, self.dim, device=device
        )
