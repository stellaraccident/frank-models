"""Activation function oracles."""

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))."""
    return x / (1 + np.exp(-x))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU activation: gate * silu(up).

    Args:
        gate: Gate tensor [batch, seq, n_ff]
        up: Up projection tensor [batch, seq, n_ff]

    Returns:
        Output tensor [batch, seq, n_ff]
    """
    return gate * silu(up)
