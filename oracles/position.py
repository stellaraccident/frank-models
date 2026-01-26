"""Position encoding oracles."""

import numpy as np


def rope(
    input: np.ndarray,
    positions: np.ndarray,
    freq_base: float = 10000.0,
    freq_scale: float = 1.0,
) -> np.ndarray:
    """Rotary Position Embeddings (RoPE).

    Applies rotation to adjacent dimension pairs based on position.
    The rotation angle for dimension pair i at position p is:
        theta = p * freq_scale / freq_base^(2i/head_dim)

    Args:
        input: Input tensor [batch, seq_len, n_head, head_dim]
        positions: Position indices [batch, seq_len]
        freq_base: Base for frequency computation (default 10000.0)
        freq_scale: Scale factor for frequencies (default 1.0)

    Returns:
        Rotated tensor [batch, seq_len, n_head, head_dim]
    """
    batch, seq_len, n_head, head_dim = input.shape
    half_dim = head_dim // 2

    # Compute frequencies: freq_scale / freq_base^(2i/head_dim)
    dim_indices = np.arange(half_dim)
    freqs = freq_scale / np.power(freq_base, 2 * dim_indices / head_dim)

    # Reshape input to [batch, seq_len, n_head, half_dim, 2]
    x = input.reshape(batch, seq_len, n_head, half_dim, 2)
    x0 = x[..., 0]  # [batch, seq_len, n_head, half_dim]
    x1 = x[..., 1]

    # Compute angles: positions[:, :, None, None] * freqs[None, None, None, :]
    # positions: [batch, seq_len] -> [batch, seq_len, 1, 1]
    # freqs: [half_dim] -> [1, 1, 1, half_dim]
    angles = positions[:, :, None, None] * freqs[None, None, None, :]

    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # Apply rotation:
    # x0' = x0*cos - x1*sin
    # x1' = x0*sin + x1*cos
    x0_rot = x0 * cos_vals - x1 * sin_vals
    x1_rot = x0 * sin_vals + x1 * cos_vals

    # Stack and reshape back to [batch, seq_len, n_head, head_dim]
    output = np.stack([x0_rot, x1_rot], axis=-1)
    return output.reshape(batch, seq_len, n_head, head_dim)
