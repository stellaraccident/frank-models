"""End-to-end tests for toy MoE LLM model.

Tests the full model by linking architecture + hparams + params + components,
compiling with fake weights, and comparing against NumPy oracles.

Toy model dimensions (for fast testing):
  vocab_size=256, n_embd=64, n_head=4, n_head_kv=2, head_dim=16
  n_ff=128, n_expert=4, n_expert_used=2, n_layers=2
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Union, List, Any

from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    MemoryType,
    VmFunction,
    VmVariantList,
)

from tests.utils import (
    link_and_compile_with_params,
    IREEConfig,
    DTYPE_TO_ELEMENT_TYPE,
)


# Paths
ARCH_DIR = Path(__file__).parent.parent.parent / "architectures"
ARCH_LLM_DIR = ARCH_DIR / "llm"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "llm"
COMPONENTS_DIR = Path(__file__).parent.parent.parent / "components"


# Toy model configuration
TOY_CONFIG = {
    "vocab_size": 256,
    "n_embd": 64,
    "n_head": 4,
    "n_head_kv": 2,
    "head_dim": 16,  # n_embd / n_head
    "n_ff": 128,
    "n_expert": 4,
    "n_expert_used": 2,
    "n_layers": 2,
    "rms_eps": 1e-5,
    "rope_freq_base": 10000.0,
}


def generate_toy_params(seed: int = 42) -> dict[str, np.ndarray]:
    """Generate random parameters for toy model matching GGUF naming convention.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict mapping parameter names to numpy arrays
    """
    rng = np.random.default_rng(seed)
    cfg = TOY_CONFIG
    params = {}

    # Model-level parameters
    params["token_embd.weight"] = rng.standard_normal(
        (cfg["vocab_size"], cfg["n_embd"])
    ).astype(np.float32)
    params["output_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(np.float32)
    params["output.weight"] = rng.standard_normal(
        (cfg["n_embd"], cfg["vocab_size"])
    ).astype(np.float32)

    # Layer-level parameters
    n_embd_kv = cfg["n_head_kv"] * cfg["head_dim"]

    for layer_idx in range(cfg["n_layers"]):
        prefix = f"blk.{layer_idx}"

        # Normalization weights
        params[f"{prefix}.attn_norm.weight"] = rng.standard_normal(
            cfg["n_embd"]
        ).astype(np.float32)
        params[f"{prefix}.ffn_norm.weight"] = rng.standard_normal(cfg["n_embd"]).astype(
            np.float32
        )

        # Attention projections
        params[f"{prefix}.attn_q.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.attn_k.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_v.weight"] = rng.standard_normal(
            (cfg["n_embd"], n_embd_kv)
        ).astype(np.float32)
        params[f"{prefix}.attn_output.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_embd"])
        ).astype(np.float32)

        # Attention biases (zeros since use_bias=false)
        params[f"{prefix}.attn_q.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)
        params[f"{prefix}.attn_k.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_v.bias"] = np.zeros(n_embd_kv, dtype=np.float32)
        params[f"{prefix}.attn_output.bias"] = np.zeros(cfg["n_embd"], dtype=np.float32)

        # MoE weights
        params[f"{prefix}.ffn_gate_inp.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_up_exps.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_ff"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_gate_exps.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_ff"], cfg["n_embd"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_down_exps.weight"] = rng.standard_normal(
            (cfg["n_expert"], cfg["n_embd"], cfg["n_ff"])
        ).astype(np.float32)

    return params


def _get_library_paths() -> list[str]:
    """Get all library paths needed for linking the toy model."""
    return [
        # Hparams and params
        str(MODELS_DIR / "toy" / "hparams.mlir"),
        str(MODELS_DIR / "toy" / "params.mlir"),
        # LLM architecture layers (under architectures/llm/)
        str(ARCH_LLM_DIR / "transformer_layer_moe_prefill.mlir"),
        str(ARCH_LLM_DIR / "transformer_layer_moe_decode.mlir"),
        # Components
        "embedding/embedding_lookup.mlir",
        "normalization/rms_norm.mlir",
        "kvcache/kvcache.mlir",
        "attention/attention_block_prefill.mlir",
        "attention/attention_block_decode.mlir",
        "attention/attention_gqa.mlir",
        "position/rope.mlir",
        "moe/moe_ffn_block.mlir",
        "moe/concat_gemm_id_silu.mlir",
        "activation/swiglu.mlir",
    ]


class ToyModelRunner:
    """Runner for toy model with support for !util.list<?> cache types.

    Extends IREEModuleWithParams to handle KV cache (VmVariantList) arguments.
    """

    def __init__(self, model):
        self._model = model
        self._device = model._device

    def _numpy_to_buffer_view(self, arr: np.ndarray) -> HalBufferView:
        arr = np.ascontiguousarray(arr)
        element_type = DTYPE_TO_ELEMENT_TYPE.get(arr.dtype.type)
        if element_type is None:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")
        return self._device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self._device,
            buffer=arr,
            element_type=element_type,
        )

    def _buffer_view_to_numpy(self, bv: HalBufferView) -> np.ndarray:
        device_array = DeviceArray(self._device, bv, implicit_host_transfer=True)
        return device_array.to_host()

    def allocate_kv_cache(self, n_blocks: int, block_size: int) -> VmVariantList:
        """Allocate KV cache, returns opaque VmVariantList."""
        func = self._model.lookup_function("allocate_kv_cache")
        arg_list = VmVariantList(2)
        arg_list.push_int(n_blocks)
        arg_list.push_int(block_size)
        result_list = VmVariantList(1)
        self._model._context.invoke(func, arg_list, result_list)
        return result_list.get_as_list(0)

    def prefill(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        start_positions: np.ndarray,
        block_size: int,
    ) -> tuple[np.ndarray, VmVariantList]:
        """Run prefill, returns (logits, updated_cache)."""
        func = self._model.lookup_function("prefill")
        arg_list = VmVariantList(6)
        arg_list.push_ref(self._numpy_to_buffer_view(tokens))
        arg_list.push_ref(self._numpy_to_buffer_view(positions))
        arg_list.push_list(cache)
        arg_list.push_ref(self._numpy_to_buffer_view(block_tables))
        arg_list.push_ref(self._numpy_to_buffer_view(start_positions))
        arg_list.push_int(block_size)

        result_list = VmVariantList(2)
        self._model._context.invoke(func, arg_list, result_list)

        logits_bv = result_list.get_as_object(0, HalBufferView)
        logits = self._buffer_view_to_numpy(logits_bv)
        cache_out = result_list.get_as_list(1)
        return logits, cache_out

    def decode(
        self,
        tokens: np.ndarray,
        positions: np.ndarray,
        cache: VmVariantList,
        block_tables: np.ndarray,
        context_lens: np.ndarray,
        max_context_len: int,
        block_indices: np.ndarray,
        pos_in_blocks: np.ndarray,
    ) -> tuple[np.ndarray, VmVariantList]:
        """Run decode step, returns (logits, updated_cache)."""
        func = self._model.lookup_function("decode")
        arg_list = VmVariantList(8)
        arg_list.push_ref(self._numpy_to_buffer_view(tokens))
        arg_list.push_ref(self._numpy_to_buffer_view(positions))
        arg_list.push_list(cache)
        arg_list.push_ref(self._numpy_to_buffer_view(block_tables))
        arg_list.push_ref(self._numpy_to_buffer_view(context_lens))
        arg_list.push_int(max_context_len)
        arg_list.push_ref(self._numpy_to_buffer_view(block_indices))
        arg_list.push_ref(self._numpy_to_buffer_view(pos_in_blocks))

        result_list = VmVariantList(2)
        self._model._context.invoke(func, arg_list, result_list)

        logits_bv = result_list.get_as_object(0, HalBufferView)
        logits = self._buffer_view_to_numpy(logits_bv)
        cache_out = result_list.get_as_list(1)
        return logits, cache_out


class TestToyModelCompilation:
    """Test that the toy model compiles successfully."""

    def test_compiles_with_params(self, iree_cfg):
        """Test that the toy model links and compiles with parameters."""
        params = generate_toy_params(seed=42)

        model = link_and_compile_with_params(
            main_path=str(ARCH_LLM_DIR / "llm_inference.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="toy_llm",
        )

        # Verify the expected functions exist
        assert model.lookup_function("allocate_kv_cache") is not None
        assert model.lookup_function("prefill") is not None
        assert model.lookup_function("decode") is not None


class TestToyModelPrefill:
    """Test prefill operation with fake weights."""

    @pytest.fixture
    def compiled_model(self, iree_cfg):
        """Compile the toy model once for all tests in this class."""
        params = generate_toy_params(seed=42)
        model = link_and_compile_with_params(
            main_path=str(ARCH_LLM_DIR / "llm_inference.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="toy_llm_prefill",
        )
        return ToyModelRunner(model)

    def test_prefill_runs(self, compiled_model):
        """Test that prefill runs without errors and produces correct shapes."""
        cfg = TOY_CONFIG
        batch = 1
        seq_len = 4
        n_layers = cfg["n_layers"]
        block_size = 16
        max_blocks_per_seq = 2
        n_blocks = batch * max_blocks_per_seq  # Total blocks in cache

        # Allocate KV cache
        cache = compiled_model.allocate_kv_cache(n_blocks, block_size)

        # Create inputs
        tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)  # [batch, seq_len]
        positions = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)

        # Block tables: [n_layers, batch, max_blocks_per_seq]
        # Map logical blocks to physical blocks (simple 1:1 mapping)
        block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[:, b, blk] = b * max_blocks_per_seq + blk

        # Start positions: [batch] - all start at position 0
        start_positions = np.zeros(batch, dtype=np.int32)

        # Run prefill
        logits, cache_out = compiled_model.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )

        # Verify output shapes
        assert logits.shape == (batch, seq_len, cfg["vocab_size"])
        assert cache_out is not None

    def test_prefill_then_decode(self, compiled_model):
        """Test prefill followed by a decode step."""
        cfg = TOY_CONFIG
        batch = 1
        prefill_len = 4
        n_layers = cfg["n_layers"]
        block_size = 16
        max_blocks_per_seq = 2
        n_blocks = batch * max_blocks_per_seq

        # Allocate KV cache
        cache = compiled_model.allocate_kv_cache(n_blocks, block_size)

        # Prefill phase
        tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
        positions = np.arange(prefill_len).reshape(1, prefill_len).astype(np.int64)
        block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[:, b, blk] = b * max_blocks_per_seq + blk
        start_positions = np.zeros(batch, dtype=np.int32)

        prefill_logits, cache = compiled_model.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )
        assert prefill_logits.shape == (batch, prefill_len, cfg["vocab_size"])

        # Decode phase - generate next token
        decode_tokens = np.array([5], dtype=np.int64)  # [batch]
        decode_positions = np.array([prefill_len], dtype=np.int64)  # [batch]

        # Context lengths after prefill: [n_layers, batch]
        context_lens = np.full((n_layers, batch), prefill_len, dtype=np.int32)
        max_context_len = prefill_len + 1  # Include new token position

        # Block indices and positions for the new token
        new_pos = prefill_len
        block_indices = np.array([new_pos // block_size], dtype=np.int32)  # [batch]
        pos_in_blocks = np.array([new_pos % block_size], dtype=np.int32)  # [batch]

        decode_logits, cache_out = compiled_model.decode(
            decode_tokens,
            decode_positions,
            cache,
            block_tables,
            context_lens,
            max_context_len,
            block_indices,
            pos_in_blocks,
        )

        # Verify decode output shape
        assert decode_logits.shape == (batch, cfg["vocab_size"])


# =============================================================================
# NumPy Oracle for Toy Model (with causal masking)
# =============================================================================


def attention_gqa_causal(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Grouped Query Attention with causal masking.

    Like attention_gqa but applies causal mask (can only attend to past/current).

    Args:
        query: Query tensor [batch, seq_len, n_head, head_dim]
        key: Key tensor [batch, seq_len, n_head_kv, head_dim]
        value: Value tensor [batch, seq_len, n_head_kv, head_dim]
        scale: Scale factor, typically 1.0 / sqrt(head_dim)

    Returns:
        Output tensor [batch, seq_len, n_head, head_dim]
    """
    batch, seq_q, n_head, head_dim = query.shape
    _, seq_kv, n_head_kv, _ = key.shape

    # GQA: Repeat KV heads to match Q heads
    repeat_factor = n_head // n_head_kv
    key = np.repeat(key, repeat_factor, axis=2)
    value = np.repeat(value, repeat_factor, axis=2)

    # Transpose for matmul: [batch, n_head, seq, head_dim]
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)

    # QK^T: [batch, n_head, seq_q, seq_kv]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask: mask positions where q_pos < k_pos
    # Create mask of shape [seq_q, seq_kv]
    q_positions = np.arange(seq_q)[:, None]  # [seq_q, 1]
    k_positions = np.arange(seq_kv)[None, :]  # [1, seq_kv]
    causal_mask = q_positions >= k_positions  # [seq_q, seq_kv] - True where can attend

    # Apply mask: set masked positions to -inf before softmax
    scores = np.where(causal_mask, scores, -np.inf)

    # Softmax (numerically stable)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_max = np.where(
        np.isinf(scores_max), 0.0, scores_max
    )  # Handle all-masked rows
    scores_exp = np.exp(scores - scores_max)
    scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
    scores_sum = np.where(scores_sum == 0, 1.0, scores_sum)  # Avoid div by zero
    attn_weights = scores_exp / scores_sum

    # Attention @ V: [batch, n_head, seq_q, head_dim]
    output = np.matmul(attn_weights, v)

    # Transpose back: [batch, seq_q, n_head, head_dim]
    return output.transpose(0, 2, 1, 3)


def attention_block_causal(
    input: np.ndarray,
    positions: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    bq: np.ndarray,
    bk: np.ndarray,
    bv: np.ndarray,
    bo: np.ndarray,
    use_bias: bool,
    n_head: int,
    n_head_kv: int,
    n_embd: int,
    rope_freq_base: float = 10000.0,
    rope_freq_scale: float = 1.0,
) -> np.ndarray:
    """Attention block with causal masking (for prefill).

    Same as attention_block from oracles but uses causal attention.
    """
    from oracles.position import rope

    batch, seq_len, _ = input.shape
    n_embd_kv = wk.shape[1]
    head_dim = n_embd // n_head
    head_dim_kv = n_embd_kv // n_head_kv

    # QKV projections
    q_proj = np.matmul(input, wq)
    k_proj = np.matmul(input, wk)
    v_proj = np.matmul(input, wv)

    if use_bias:
        q_proj = q_proj + bq
        k_proj = k_proj + bk
        v_proj = v_proj + bv

    # Reshape for multi-head
    q_reshaped = q_proj.reshape(batch, seq_len, n_head, head_dim)
    k_reshaped = k_proj.reshape(batch, seq_len, n_head_kv, head_dim_kv)
    v_reshaped = v_proj.reshape(batch, seq_len, n_head_kv, head_dim_kv)

    # Apply RoPE
    q_rope = rope(q_reshaped, positions, rope_freq_base, rope_freq_scale)
    k_rope = rope(k_reshaped, positions, rope_freq_base, rope_freq_scale)

    # Compute attention with causal masking
    scale = 1.0 / np.sqrt(head_dim)
    attn_out = attention_gqa_causal(q_rope, k_rope, v_reshaped, scale)

    # Reshape and output projection
    attn_flat = attn_out.reshape(batch, seq_len, n_embd)
    output_proj = np.matmul(attn_flat, wo)

    if use_bias:
        output_proj = output_proj + bo

    return output_proj


def transformer_layer_moe_causal(
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
    """MoE transformer layer with causal attention (for prefill).

    Same as transformer_layer_moe but uses causal attention.
    """
    from oracles.normalization import rms_norm
    from oracles.moe import moe_ffn_block

    batch, seq_len, _ = input.shape
    idx = layer_idx

    # Extract parameters
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

    # Attention sub-layer
    input_2d = input.reshape(batch * seq_len, n_embd)
    attn_normed = rms_norm(input_2d, attn_norm_w, rms_eps)
    attn_normed_3d = attn_normed.reshape(batch, seq_len, n_embd)

    attn_out = attention_block_causal(
        attn_normed_3d,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )

    residual1 = input + attn_out

    # MoE FFN sub-layer
    residual1_2d = residual1.reshape(batch * seq_len, n_embd)
    ffn_normed = rms_norm(residual1_2d, ffn_norm_w, rms_eps)

    moe_out = moe_ffn_block(
        ffn_normed,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    moe_out_3d = moe_out.reshape(batch, seq_len, n_embd)
    output = residual1 + moe_out_3d
    return output


def prefill_oracle(
    tokens: np.ndarray,
    positions: np.ndarray,
    params: dict[str, np.ndarray],
    cfg: dict,
) -> np.ndarray:
    """NumPy oracle for prefill forward pass.

    Note: This uses non-causal attention to match the current IREE
    attention_gqa component, which doesn't apply causal masking.
    For a proper autoregressive model, causal masking should be added
    to the IREE attention component and this oracle updated to match.
    See: https://github.com/stellaraccident/frank-models/issues/4

    Args:
        tokens: Token indices [batch, seq_len]
        positions: Position indices [batch, seq_len]
        params: Model parameters dict
        cfg: Model configuration dict

    Returns:
        Logits [batch, seq_len, vocab_size]
    """
    from oracles.embedding import embedding_lookup
    from oracles.normalization import rms_norm
    from oracles.transformer_layer import transformer_layer_moe

    batch, seq_len = tokens.shape

    # Token embedding lookup
    embeddings = embedding_lookup(params["token_embd.weight"], tokens)

    # Transformer layers (non-causal attention matches IREE component)
    hidden = embeddings
    for layer_idx in range(cfg["n_layers"]):
        hidden = transformer_layer_moe(
            hidden,
            positions,
            params,
            layer_idx=layer_idx,
            n_head=cfg["n_head"],
            n_head_kv=cfg["n_head_kv"],
            n_embd=cfg["n_embd"],
            n_ff=cfg["n_ff"],
            n_expert=cfg["n_expert"],
            n_expert_used=cfg["n_expert_used"],
            rms_eps=cfg["rms_eps"],
            rope_freq_base=cfg["rope_freq_base"],
            rope_freq_scale=1.0,
            use_bias=False,
            normalize_weights=False,
        )

    # Output normalization
    hidden_2d = hidden.reshape(batch * seq_len, cfg["n_embd"])
    normalized = rms_norm(hidden_2d, params["output_norm.weight"], cfg["rms_eps"])
    normalized = normalized.reshape(batch, seq_len, cfg["n_embd"])

    # LM head projection
    logits = np.matmul(normalized, params["output.weight"])

    return logits


class TestToyModelNumeric:
    """Numeric validation tests comparing IREE output to NumPy oracle."""

    @pytest.fixture
    def model_and_params(self, iree_cfg):
        """Compile model and return (runner, params)."""
        params = generate_toy_params(seed=42)
        model = link_and_compile_with_params(
            main_path=str(ARCH_LLM_DIR / "llm_inference.mlir"),
            library_paths=_get_library_paths(),
            iree_cfg=iree_cfg,
            params=params,
            scope="model",
            debug_name="toy_llm_numeric",
        )
        return ToyModelRunner(model), params

    def test_prefill_matches_oracle(self, model_and_params):
        """Test that prefill output matches NumPy oracle."""
        runner, params = model_and_params
        cfg = TOY_CONFIG
        batch = 1
        seq_len = 4
        n_layers = cfg["n_layers"]
        block_size = 16
        max_blocks_per_seq = 2
        n_blocks = batch * max_blocks_per_seq

        # Allocate KV cache
        cache = runner.allocate_kv_cache(n_blocks, block_size)

        # Create inputs
        tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
        positions = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)
        block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[:, b, blk] = b * max_blocks_per_seq + blk
        start_positions = np.zeros(batch, dtype=np.int32)

        # Run IREE model
        iree_logits, _ = runner.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )

        # Run NumPy oracle
        oracle_logits = prefill_oracle(tokens, positions, params, cfg)

        # Compare - use slightly relaxed tolerance for MoE transformer
        # (complex computation with many ops leads to small numerical divergence)
        np.testing.assert_allclose(
            iree_logits,
            oracle_logits,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Prefill logits don't match oracle",
        )

    def test_prefill_batch2(self, model_and_params):
        """Test prefill with batch size 2."""
        runner, params = model_and_params
        cfg = TOY_CONFIG
        batch = 2
        seq_len = 3
        n_layers = cfg["n_layers"]
        block_size = 16
        max_blocks_per_seq = 2
        n_blocks = batch * max_blocks_per_seq

        # Allocate KV cache
        cache = runner.allocate_kv_cache(n_blocks, block_size)

        # Create inputs - different tokens per batch
        tokens = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
        positions = (
            np.arange(seq_len)
            .reshape(1, seq_len)
            .repeat(batch, axis=0)
            .astype(np.int64)
        )
        block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[:, b, blk] = b * max_blocks_per_seq + blk
        start_positions = np.zeros(batch, dtype=np.int32)

        # Run IREE model
        iree_logits, _ = runner.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )

        # Run NumPy oracle
        oracle_logits = prefill_oracle(tokens, positions, params, cfg)

        # Compare
        np.testing.assert_allclose(
            iree_logits,
            oracle_logits,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Batch-2 prefill logits don't match oracle",
        )

    def test_prefill_longer_sequence(self, model_and_params):
        """Test prefill with longer sequence."""
        runner, params = model_and_params
        cfg = TOY_CONFIG
        batch = 1
        seq_len = 8
        n_layers = cfg["n_layers"]
        block_size = 16
        max_blocks_per_seq = 2
        n_blocks = batch * max_blocks_per_seq

        # Allocate KV cache
        cache = runner.allocate_kv_cache(n_blocks, block_size)

        # Create inputs
        tokens = np.arange(1, seq_len + 1).reshape(1, seq_len).astype(np.int64)
        positions = np.arange(seq_len).reshape(1, seq_len).astype(np.int64)
        block_tables = np.zeros((n_layers, batch, max_blocks_per_seq), dtype=np.int32)
        for b in range(batch):
            for blk in range(max_blocks_per_seq):
                block_tables[:, b, blk] = b * max_blocks_per_seq + blk
        start_positions = np.zeros(batch, dtype=np.int32)

        # Run IREE model
        iree_logits, _ = runner.prefill(
            tokens, positions, cache, block_tables, start_positions, block_size
        )

        # Run NumPy oracle
        oracle_logits = prefill_oracle(tokens, positions, params, cfg)

        # Compare
        np.testing.assert_allclose(
            iree_logits,
            oracle_logits,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Longer sequence prefill logits don't match oracle",
        )
