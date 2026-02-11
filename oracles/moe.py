"""NumPy oracles for MoE (Mixture of Experts) components.

Reference implementations matching MLIR semantics in components/moe/.
"""

import numpy as np


def moe_ffn_block(
    input: np.ndarray,
    gate_inp_w: np.ndarray,
    up_exps_w: np.ndarray,
    gate_exps_w: np.ndarray,
    down_exps_w: np.ndarray,
    n_expert: int,
    n_expert_used: int,
    n_embd: int,
    n_ff: int,
    normalize_weights: bool,
) -> np.ndarray:
    """Full MoE FFN block.

    Args:
        input: Token embeddings [n_tokens, n_embd]
        gate_inp_w: Router weights [n_expert, n_embd]
        up_exps_w: Expert up projections [n_expert, n_ff, n_embd]
        gate_exps_w: Expert gate projections [n_expert, n_ff, n_embd]
        down_exps_w: Expert down projections [n_expert, n_embd, n_ff]
        n_expert: Total number of experts
        n_expert_used: Top-k experts per token
        n_embd: Embedding dimension
        n_ff: Feed-forward hidden dimension
        normalize_weights: Whether to renormalize top-k weights

    Returns:
        Output [n_tokens, n_embd]

    Reference: components/moe/moe_ffn_block.mlir
    """
    from oracles.activation import swiglu

    # Step 1: Router logits [n_expert, n_tokens]
    logits = gate_inp_w @ input.T

    # Step 2: Softmax over experts (axis 0)
    exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    # Step 3: Top-k selection
    # argsort descending, take top k
    top_indices = np.argsort(-probs, axis=0)[
        :n_expert_used
    ]  # [n_expert_used, n_tokens]
    top_weights = np.take_along_axis(
        probs, top_indices, axis=0
    )  # [n_expert_used, n_tokens]

    # Step 4: Optional normalization
    if normalize_weights:
        weight_sums = top_weights.sum(axis=0, keepdims=True)
        top_weights = top_weights / weight_sums

    # Step 5: Fused UP + GATE projection (single gather + matmul)
    up_gate_w = np.concatenate([up_exps_w, gate_exps_w], axis=1)
    gathered_up_gate = up_gate_w[top_indices.astype(np.int32)]
    up_gate = np.einsum("etoi,it->oet", gathered_up_gate, input.T)
    up, gate = up_gate[:n_ff], up_gate[n_ff:]

    # Step 7: SwiGLU activation
    activated = swiglu(gate, up)

    # Step 8: Expert DOWN projection
    gathered_down = down_exps_w[top_indices.astype(np.int32)]
    experts_out = np.einsum("etoi,iet->oet", gathered_down, activated)

    # Step 9: Apply weights [n_embd, n_expert_used, n_tokens] * [n_expert_used, n_tokens]
    experts_weighted = experts_out * top_weights[None, :, :]

    # Step 10: Sum over expert dimension
    output_summed = experts_weighted.sum(axis=1)  # [n_embd, n_tokens]

    # Transpose to [n_tokens, n_embd]
    return output_summed.T
