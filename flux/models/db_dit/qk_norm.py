"""QK normalization utilities (re-exported for convenience).

The actual QK normalization is applied within the MultiHeadAttention module.
This module provides the standalone norm layers if needed elsewhere.
"""

from flux.models.common.norm import RMSNorm

QKLayerNorm = RMSNorm  # Alias for clarity
