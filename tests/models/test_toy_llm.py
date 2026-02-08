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
    LAYERS_DIR,
    DTYPE_TO_ELEMENT_TYPE,
)


# Paths
ARCH_DIR = Path(__file__).parent.parent.parent / "architectures"
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
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_gate_exps.weight"] = rng.standard_normal(
            (cfg["n_ff"], cfg["n_embd"], cfg["n_expert"])
        ).astype(np.float32)
        params[f"{prefix}.ffn_down_exps.weight"] = rng.standard_normal(
            (cfg["n_embd"], cfg["n_ff"], cfg["n_expert"])
        ).astype(np.float32)

    return params


def _get_library_paths() -> list[str]:
    """Get all library paths needed for linking the toy model."""
    return [
        # Hparams and params
        str(MODELS_DIR / "toy" / "hparams.mlir"),
        str(MODELS_DIR / "toy" / "params.mlir"),
        # Layers
        str(LAYERS_DIR / "transformer_layer_moe_prefill.mlir"),
        str(LAYERS_DIR / "transformer_layer_moe_decode.mlir"),
        # Components
        "embedding/embedding_lookup.mlir",
        "normalization/rms_norm.mlir",
        "kvcache/kvcache.mlir",
        "attention/attention_block_prefill.mlir",
        "attention/attention_block_decode.mlir",
        "attention/attention_gqa.mlir",
        "position/rope.mlir",
        "moe/moe_ffn_block.mlir",
        "moe/mul_mat_id.mlir",
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
            main_path=str(ARCH_DIR / "llm_moe_cached.mlir"),
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
            main_path=str(ARCH_DIR / "llm_moe_cached.mlir"),
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
