"""NumPy reference implementations for normalization operations.

These implementations must match the MLIR semantics in components/normalization/
exactly. They serve as oracles for validating IREE compilation.
"""

import numpy as np


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS Normalization.

    Computes: out = (x / sqrt(mean(x^2, axis=-1) + eps)) * weight

    This matches the semantics of @rms_norm_linalg in
    components/normalization/rms_norm.mlir

    Args:
        x: Input tensor of shape [..., hidden_dim]
        weight: Scale weights of shape [hidden_dim]
        eps: Small constant for numerical stability

    Returns:
        Normalized and scaled tensor, same shape as input

    Algorithm (matching MLIR):
        1. Compute sum of squares: sum_sq = sum(x^2, axis=-1)
        2. Compute mean: mean_sq = sum_sq / hidden_dim
        3. Compute RMS: rms = sqrt(mean_sq + eps)
        4. Normalize: normalized = x / rms
        5. Scale: output = normalized * weight
    """
    # Match MLIR: reduction over last axis
    sum_sq = np.sum(x * x, axis=-1, keepdims=True)
    hidden_dim = x.shape[-1]
    mean_sq = sum_sq / hidden_dim
    rms = np.sqrt(mean_sq + eps)
    normalized = x / rms
    return normalized * weight


def layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer Normalization.

    Computes: out = ((x - mean) / sqrt(var + eps)) * weight + bias

    Args:
        x: Input tensor of shape [..., hidden_dim]
        weight: Scale weights of shape [hidden_dim]
        bias: Bias of shape [hidden_dim]
        eps: Small constant for numerical stability

    Returns:
        Normalized, scaled, and shifted tensor
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return normalized * weight + bias
