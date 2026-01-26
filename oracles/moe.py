"""NumPy oracles for MoE (Mixture of Experts) components.

Reference implementations matching MLIR semantics in components/moe/.
"""

import numpy as np


def mul_mat_id(
    weights: np.ndarray,
    input: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    """Expert-selected matrix multiply.

    Dynamically selects expert weight matrices based on indices and performs
    batched matrix multiplication. Core primitive for MoE layers.

    Args:
        weights: Expert weight matrices [n_out, n_in, n_expert]
        input: Input activations [n_in, n_expert_used, n_tokens]
        ids: Expert indices [n_expert_used, n_tokens] (int32)

    Returns:
        Output [n_out, n_expert_used, n_tokens]

    Reference: components/moe/mul_mat_id.mlir
    """
    n_out, n_in, n_expert = weights.shape
    n_expert_used, n_tokens = ids.shape

    # Transpose weights for easier indexing: [n_expert, n_out, n_in]
    weights_t = weights.transpose(2, 0, 1)

    # Gather: select expert matrices based on ids
    # gathered shape: [n_expert_used, n_tokens, n_out, n_in]
    gathered = weights_t[ids]

    # Reshape input: [n_in, n_expert_used, n_tokens] -> [n_expert_used, n_tokens, n_in]
    input_t = input.transpose(1, 2, 0)

    # Batched matmul per (expert_used, token) pair
    # gathered[e,t] @ input_t[e,t] for each e, t
    # Using einsum: 'etoi,eti->eto' (matrix-vector product)
    output = np.einsum("etoi,eti->eto", gathered, input_t)

    # Transpose to [n_out, n_expert_used, n_tokens]
    return output.transpose(2, 0, 1)


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
        up_exps_w: Expert up projections [n_ff, n_embd, n_expert]
        gate_exps_w: Expert gate projections [n_ff, n_embd, n_expert]
        down_exps_w: Expert down projections [n_embd, n_ff, n_expert]
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

    n_tokens = input.shape[0]

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

    # Step 5: Replicate input for each expert slot
    # input is [n_tokens, n_embd], we want [n_embd, n_expert_used, n_tokens]
    input_replicated = np.broadcast_to(
        input.T[:, None, :], (n_embd, n_expert_used, n_tokens)  # [n_embd, 1, n_tokens]
    ).copy()

    # Step 6: Expert UP projection
    up = mul_mat_id(up_exps_w, input_replicated, top_indices.astype(np.int32))

    # Step 7: Expert GATE projection
    gate = mul_mat_id(gate_exps_w, input_replicated, top_indices.astype(np.int32))

    # Step 8: SwiGLU activation
    activated = swiglu(gate, up)

    # Step 9: Expert DOWN projection
    experts_out = mul_mat_id(down_exps_w, activated, top_indices.astype(np.int32))

    # Step 10: Apply weights [n_embd, n_expert_used, n_tokens] * [n_expert_used, n_tokens]
    experts_weighted = experts_out * top_weights[None, :, :]

    # Step 11: Sum over expert dimension
    output_summed = experts_weighted.sum(axis=1)  # [n_embd, n_tokens]

    # Transpose to [n_tokens, n_embd]
    return output_summed.T
