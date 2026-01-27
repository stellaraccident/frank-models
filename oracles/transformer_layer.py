"""NumPy oracle for MoE transformer layer.

Composes existing oracles: rms_norm, attention_block, moe_ffn_block.
Matches the MLIR semantics in layers/transformer_layer_moe.mlir.
"""

import numpy as np

from oracles.normalization import rms_norm
from oracles.attention import attention_block
from oracles.moe import moe_ffn_block


def transformer_layer_moe(
    input: np.ndarray,
    positions: np.ndarray,
    params: dict[str, np.ndarray],
    layer_idx: int,
    n_head: int,
    n_head_kv: int,
    n_embd: int,
    n_ff: int,
    n_expert: int,
    n_expert_used: int,
    rms_eps: float = 1e-5,
    rope_freq_base: float = 10000.0,
    rope_freq_scale: float = 1.0,
    use_bias: bool = False,
    normalize_weights: bool = False,
) -> np.ndarray:
    """Full MoE transformer layer: pre-norm attention + MoE FFN with residuals.

    Args:
        input: [batch, seq_len, n_embd]
        positions: [batch, seq_len] int64
        params: Dict mapping GGUF-style names to numpy arrays
        layer_idx: Layer index for parameter lookup
        n_head: Number of query attention heads
        n_head_kv: Number of key/value attention heads
        n_embd: Embedding dimension
        n_ff: Feed-forward hidden dimension
        n_expert: Total number of experts
        n_expert_used: Top-k experts per token
        rms_eps: RMS norm epsilon
        rope_freq_base: RoPE frequency base
        rope_freq_scale: RoPE frequency scale
        use_bias: Whether attention uses biases
        normalize_weights: Whether to renormalize MoE weights

    Returns:
        Output [batch, seq_len, n_embd]
    """
    batch, seq_len, _ = input.shape
    idx = layer_idx

    # Extract parameters.
    attn_norm_w = params[f"blk.{idx}.attn_norm.weight"]
    ffn_norm_w = params[f"blk.{idx}.ffn_norm.weight"]
    wq = params[f"blk.{idx}.attn_q.weight"]
    wk = params[f"blk.{idx}.attn_k.weight"]
    wv = params[f"blk.{idx}.attn_v.weight"]
    wo = params[f"blk.{idx}.attn_output.weight"]
    bq = params[f"blk.{idx}.attn_q.bias"]
    bk = params[f"blk.{idx}.attn_k.bias"]
    bv = params[f"blk.{idx}.attn_v.bias"]
    bo = params[f"blk.{idx}.attn_output.bias"]
    gate_inp_w = params[f"blk.{idx}.ffn_gate_inp.weight"]
    up_exps_w = params[f"blk.{idx}.ffn_up_exps.weight"]
    gate_exps_w = params[f"blk.{idx}.ffn_gate_exps.weight"]
    down_exps_w = params[f"blk.{idx}.ffn_down_exps.weight"]

    # ---- Attention sub-layer ----
    # Flatten [batch, seq_len, n_embd] → [batch*seq_len, n_embd] for rms_norm.
    input_2d = input.reshape(batch * seq_len, n_embd)
    attn_normed = rms_norm(input_2d, attn_norm_w, rms_eps)
    # Unflatten back to [batch, seq_len, n_embd] for attention_block.
    attn_normed_3d = attn_normed.reshape(batch, seq_len, n_embd)

    attn_out = attention_block(
        attn_normed_3d, positions,
        wq, wk, wv, wo,
        bq, bk, bv, bo,
        use_bias, n_head, n_head_kv, n_embd,
        rope_freq_base, rope_freq_scale,
    )

    # Residual connection.
    residual1 = input + attn_out

    # ---- MoE FFN sub-layer ----
    # Flatten [batch, seq_len, n_embd] → [n_tokens, n_embd].
    residual1_2d = residual1.reshape(batch * seq_len, n_embd)
    ffn_normed = rms_norm(residual1_2d, ffn_norm_w, rms_eps)

    moe_out = moe_ffn_block(
        ffn_normed, gate_inp_w,
        up_exps_w, gate_exps_w, down_exps_w,
        n_expert, n_expert_used, n_embd, n_ff,
        normalize_weights,
    )

    # Unflatten [n_tokens, n_embd] → [batch, seq_len, n_embd].
    moe_out_3d = moe_out.reshape(batch, seq_len, n_embd)

    # Residual connection.
    output = residual1 + moe_out_3d
    return output
